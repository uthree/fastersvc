import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from module.dataset import WaveFileDirectory
from module.loss import MultiScaleSTFTLoss, LogMelSpectrogramLoss
from module.pitch_estimator import PitchEstimator
from module.content_encoder import ContentEncoder
from module.speaker_encoder import SpeakerEncoder
from module.decoder import Decoder
from module.common import energy
from module.discriminator import Discriminator


parser = argparse.ArgumentParser(description="train voice conversion model")

parser.add_argument('dataset')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-pep', '--pitch-estimator-path', default='models/pitch_estimator.pt')
parser.add_argument('-sep', '--speaker-encoder-path', default='models/speaker_encoder.pt')
parser.add_argument('-dip', '--discriminator-path', default='models/discriminator.pt')
parser.add_argument('-dep', '--decoder-path', default='models/decoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-len', '--length', default=48000, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-no-spk', default=False, type=bool)

parser.add_argument('--weight-adv', default=0.2, type=float)
parser.add_argument('--weight-mel', default=1.0, type=float)

args = parser.parse_args()

WEIGHT_ADV = args.weight_adv
WEIGHT_MEL = args.weight_mel

def load_or_init_models(device=torch.device('cpu')):
    dec = Decoder().to(device)
    se = SpeakerEncoder().to(device)
    dis = Discriminator().to(device)
    if os.path.exists(args.decoder_path):
        dec.load_state_dict(torch.load(args.decoder_path, map_location=device))
    if os.path.exists(args.speaker_encoder_path):
        se.load_state_dict(torch.load(args.speaker_encoder_path, map_location=device))
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    return dec, se, dis


def save_models(dec, se, dis):
    print("Saving models...")
    torch.save(dec.state_dict(), args.decoder_path)
    torch.save(se.state_dict(), args.speaker_encoder_path)
    torch.save(dis.state_dict(), args.discriminator_path)
    print("Complete!")


def cut_center(x):
    length = x.shape[1]
    center = length // 2
    size = length // 2
    return x[:, center-size:center+size]

device = torch.device(args.device)

Dec, SE, Dis = load_or_init_models(device)
PE = PitchEstimator().to(device).eval()
PE.load_state_dict(torch.load(args.pitch_estimator_path, map_location=device))
CE = ContentEncoder().to(device).eval()
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptDec = optim.AdamW(Dec.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptSE = optim.AdamW(SE.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptDis = optim.AdamW(Dis.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))

stft_loss = MultiScaleSTFTLoss().to(device)
logmel_loss = LogMelSpectrogramLoss().to(device)

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device)
        N = wave.shape[0]
        
        # train generator and speaker encoder
        OptDec.zero_grad()
        OptSE.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            z = CE.encode_train(wave)
            if args.no_spk:
                spk = torch.zeros(N, 256, 1, device=device)
            else:
                spk = SE.encode(wave)
            p = PE.estimate(wave)
            e = energy(wave)
            fake = Dec.synthesize(z, p, e, spk)

            # remove nan
            fake[fake.isnan()] = 0

            loss_adv = 0
            for logit in Dis.logits(cut_center(fake)):
                logit[logit.isnan()] = 0
                loss_adv += (logit ** 2).mean()
            loss_stft = stft_loss(fake, wave)
            loss_mel = logmel_loss(fake, wave)
            loss_g = loss_stft + loss_adv * WEIGHT_ADV + loss_mel * WEIGHT_MEL

        scaler.scale(loss_g).backward()
        scaler.step(OptDec)
        if not args.no_spk:
            scaler.step(OptSE)

        # train discriminator
        fake = fake.detach()
        OptDis.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_d = 0
            for logit in Dis.logits(cut_center(wave)):
                logit[logit.isnan()] = 0
                loss_d += (logit ** 2).mean()
            for logit in Dis.logits(cut_center(fake)):
                logit[logit.isnan()] = 1
                loss_d += ((logit - 1) ** 2).mean()

        scaler.scale(loss_d).backward()
        scaler.step(OptDis)

        scaler.update()

        step_count += 1
        
        tqdm.write(f"Step {step_count}, D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Mel.: {loss_mel.item():.4f}, STFT: {loss_stft.item():.4f}")

        bar.update(N)

        if batch % 200 == 0:
            save_models(Dec, SE, Dis)

print("Training Complete!")
save_models(Dec, SE, Dis)
