import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from module.dataset import Dataset
from module.loss import MultiScaleSTFTLoss
from module.pitch_estimator import PitchEstimator
from module.content_encoder import ContentEncoder
from module.decoder import Decoder
from module.common import energy, match_features
from module.discriminator import Discriminator


parser = argparse.ArgumentParser(description="train voice conversion model")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-pep', '--pitch-estimator-path', default='models/pitch_estimator.pt')
parser.add_argument('-dip', '--discriminator-path', default='models/discriminator.pt')
parser.add_argument('-step', '--max-steps', default=300000, type=int)
parser.add_argument('-join-d', '--discriminator-join-steps', default=100000, type=int)
parser.add_argument('-dep', '--decoder-path', default='models/decoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=10000, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--save-interval', default=100, type=int)
parser.add_argument('-fp16', default=False, type=bool)

parser.add_argument('--weight-adv', default=2.5, type=float)
parser.add_argument('--weight-stft', default=1.0, type=float)

args = parser.parse_args()

WEIGHT_ADV = args.weight_adv
WEIGHT_STFT = args.weight_stft

def load_or_init_models(device=torch.device('cpu')):
    dec = Decoder().to(device)
    dis = Discriminator().to(device)
    if os.path.exists(args.decoder_path):
        dec.load_state_dict(torch.load(args.decoder_path, map_location=device))
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    return dec, dis


def save_models(dec, dis):
    print("Saving models...")
    torch.save(dec.state_dict(), args.decoder_path)
    torch.save(dis.state_dict(), args.discriminator_path)
    print("Complete!")


def center(wave, length=16000):
    c = wave.shape[1] // 2
    half_len = length // 2
    return wave[:, c-half_len:c+half_len]


device = torch.device(args.device)

Dec, Dis = load_or_init_models(device)
PE = PitchEstimator().to(device).eval()
PE.load_state_dict(torch.load(args.pitch_estimator_path, map_location=device))
CE = ContentEncoder().to(device).eval()
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))

ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptDec = optim.AdamW(Dec.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptDis = optim.AdamW(Dis.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))

multiscale_stft_loss = MultiScaleSTFTLoss().to(device)

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        N = wave.shape[0]
        discriminator_join = step_count > args.discriminator_join_steps

        # train generator and speaker encoder
        OptDec.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wave = wave.to(device)
            wave = (wave / wave.abs().max(dim=1, keepdim=True).values) * torch.rand(N, 1, device=device)
            f0 = f0.to(device)

            z = CE.encode(wave)
            e = energy(wave)
            z = match_features(z, z).detach()
            fake = Dec.synthesize(z, f0, e)

            fake[fake.isnan()] = 0
            fake[fake.isinf()] = 0

            loss_stft = multiscale_stft_loss(fake, wave)

            if discriminator_join:
                loss_adv = 0
                logits, _ = Dis(center(fake))
                for logit in logits:
                    logit[logit.isnan()] = 0
                    loss_adv += (logit ** 2).mean() / len(logits)
                loss_g = loss_adv * WEIGHT_ADV + loss_stft * WEIGHT_STFT
            else:
                loss_g = loss_stft

        scaler.scale(loss_g).backward()
        nn.utils.clip_grad_norm_(Dec.parameters(), 1.0)
        scaler.step(OptDec)
        
        if discriminator_join:
            # train discriminator
            fake = fake.detach()
            fake = fake.clamp(-1.0, 1.0)

            OptDis.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                loss_d = 0
                logits, _ = Dis(center(wave))
                for logit in logits:
                    loss_d += (logit ** 2).mean() / len(logits)
                logits, _ = Dis(center(fake))
                for logit in logits:
                    loss_d += ((logit - 1) ** 2).mean() / len(logits)

            scaler.scale(loss_d).backward()
            nn.utils.clip_grad_norm_(Dis.parameters(), 1.0)
            scaler.step(OptDis)

        scaler.update()

        step_count += 1
        
        if discriminator_join:
            tqdm.write(f"Epoch {epoch}, Step {step_count}, Dis.: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, STFT.: {loss_stft.item():.4f}")
        else:
            tqdm.write(f"Epoch {epoch}, Step {step_count}, STFT.: {loss_stft.item():.4f}")
            
        bar.update(N)

        if batch % args.save_interval == 0:
            save_models(Dec, Dis)
    if step_count >= args.max_steps:
        break

print("Training Complete!")
save_models(Dec, Dis)
