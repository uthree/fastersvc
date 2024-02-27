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
from transformers import HubertForCTC

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

hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
hubert.eval()

CE = load_or_init_models(device)
ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
Opt = optim.RAdam(CE.parameters(), lr=args.learning_rate)
cross_entropy_loss = nn.CrossEntropyLoss()

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)

        with torch.no_grad():
            wave_16k = resample(wave, 24000, 16000)
            pseudo_label = hubert(wave_16k).logits.argmax(dim=2)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            out = CE.encode(wave)
            pred = F.interpolate(CE.to_hubert(out), pseudo_label.shape[1])
            loss = cross_entropy_loss(pred, pseudo_label)

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
