import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.dataset import WaveFileDirectory
from module.content_encoder import ContentEncoder
from transformers import HubertForCTC

parser = argparse.ArgumentParser(description="distillation of hubert")

parser.add_argument('dataset')
parser.add_argument('--hubert', default='facebook/hubert-large-ls960-ft')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-len', '--length', default=122880, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
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

CE = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

Opt = optim.RAdam(CE.parameters(), lr=args.learning_rate)

hubert = HubertForCTC.from_pretrained(args.hubert).to(device).eval()

cross_entropy = nn.CrossEntropyLoss()

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            wave_16k = resample(wave, 48000, 16000)
            with torch.no_grad():
                hubert_features = hubert(wave_16k).logits
                hubert_features = hubert_features.transpose(1, 2)

        Opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            z = CE.encode(wave, softmax=False)
            labels = F.interpolate(hubert_features, z.shape[2]).argmax(dim=1)
            loss = cross_entropy(z, labels)

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
