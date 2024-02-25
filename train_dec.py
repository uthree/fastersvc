import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from module.dataset import Dataset
from module.loss import LogMelSpectrogramLoss
from module.pitch_estimator import PitchEstimator
from module.content_encoder import ContentEncoder
from module.decoder import Decoder
from module.common import energy, instance_norm
from module.speaker_embedding import SpeakerEmbedding
from module.discriminator import Discriminator


parser = argparse.ArgumentParser(description="train voice conversion model")

parser.add_argument('-dataset-cache', default='./dataset_cache')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-dip', '--discriminator-path', default='models/discriminator.pt')
parser.add_argument('-sep', '--speaker-embedding-path', default='models/speaker_embedding.pt')
parser.add_argument('-dep', '--decoder-path', default='models/decoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--save-interval', type=int, default=100)
parser.add_argument('-fp16', default=False, type=bool)

parser.add_argument('--weight-adv', default=0.0, type=float)
parser.add_argument('--weight-feat', default=2.0, type=float)
parser.add_argument('--weight-mel', default=1.0, type=float)

args = parser.parse_args()

WEIGHT_FEAT = args.weight_feat
WEIGHT_ADV = args.weight_adv
WEIGHT_MEL = args.weight_mel

def load_or_init_models(device=torch.device('cpu')):
    dec = Decoder().to(device)
    dis = Discriminator().to(device)
    se = SpeakerEmbedding().to(device)
    if os.path.exists(args.decoder_path):
        dec.load_state_dict(torch.load(args.decoder_path, map_location=device))
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    if os.path.exists(args.speaker_embedding_path):
        se.load_state_dict(torch.load(args.speaker_embedding_path, map_location=device))
    return dec, dis, se


def save_models(dec, dis, se):
    print("Saving models...")
    torch.save(dec.state_dict(), args.decoder_path)
    torch.save(dis.state_dict(), args.discriminator_path)
    torch.save(se.state_dict(), args.speaker_embedding_path)
    print("Complete!")


def center(wave, length=16000):
    c = wave.shape[1] // 2
    half_len = length // 2
    return wave[:, c-half_len:c+half_len]


device = torch.device(args.device)

CE = ContentEncoder().to(device)
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
Dec, Dis, SE = load_or_init_models(device)

ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptDec = optim.AdamW(Dec.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptDis = optim.AdamW(Dis.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))

logmel_loss = LogMelSpectrogramLoss().to(device)

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)
        # Data argumentation
        wave = wave / wave.abs().max(dim=1, keepdim=True).values * torch.rand(N, 1, device=device)
        f0 = f0.to(device)
        spk_id = spk_id.to(device)

        # train generator and speaker encoder
        OptDec.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):

            z = CE.encode(wave)
            z = instance_norm(z)
            e = energy(wave)
            spk = SE(spk_id)
            fake = Dec.synthesize(z, f0, e, spk)

            loss_adv = 0
            loss_feat = 0
            logits, feats_fake = Dis(center(fake))
            loss_mel = logmel_loss(fake, wave)

            _, feats_real = Dis(center(wave))
            for logit in logits:
                loss_adv += (logit ** 2).mean() / len(logits)
            for f, r in zip(feats_fake, feats_real):
                loss_feat += (f - r).abs().mean() / len(feats_fake)

            loss_norm = (fake.mean(dim=1) ** 2).mean()
            loss_g = loss_adv * WEIGHT_ADV + loss_feat * WEIGHT_FEAT + loss_mel * WEIGHT_MEL + loss_norm

        scaler.scale(loss_g).backward()
        nn.utils.clip_grad_norm_(Dec.parameters(), 1.0)
        scaler.step(OptDec)

        # train discriminator
        fake = fake.detach()
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
        
        tqdm.write(f"Epoch {epoch}, Step {step_count}, Dis.: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Feat.: {loss_feat.item():.4f}, Mel.: {loss_mel.item():.4f}")

        bar.update(N)

        if batch % args.save_interval == 0:
            save_models(Dec, Dis, SE)

print("Training Complete!")
save_models(Dec, Dis, SE)
