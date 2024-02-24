import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.dataset import Dataset
from module.content_encoder import ContentEncoder
from transformers import HubertModel

parser = argparse.ArgumentParser(description="distillation of hubert")

parser.add_argument('-dataset-cache', default='./dataset_cache')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-fp16', default=False, type=bool)

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    ce = ContentEncoder().to(device)
    if os.path.exists(args.content_encoder_path):
        ce.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
    return ce

def save_models(ce):
    print("Saving models...")
    torch.save(ce.state_dict(), args.content_encoder_path)
    print("Complete!")

device = torch.device(args.device)

hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
hubert = hubert.to(device)

CE = load_or_init_models(device)
ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
Opt = optim.RAdam(CE.parameters(), lr=args.learning_rate)

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)

        hubert_features = hubert.units(resample(wave.unsqueeze(1), 24000, 16000)).transpose(1, 2)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            out = CE.encode(wave)
            pred_feat = CE.to_hubert(out)
            loss = (F.interpolate(hubert_features, pred_feat.shape[2]) - pred_feat).abs().mean()

        scaler.scale(loss).backward()
        scaler.step(Opt)

        scaler.update()

        step_count += 1

        tqdm.write(f"Step {step_count}, loss: {loss.item()}")

        bar.update(N)

        if batch % 500 == 0:
            save_models(CE)

print("Training Complete!")
save_models(CE)
