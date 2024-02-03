import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from module.dataset import WaveFileDirectory
from module.content_encoder import ContentEncoder

def shuffle(tensor, dim):
    indices = torch.randperm(tensor.size(dim))
    shuffled_tensor = tensor.index_select(dim, indices)
    return shuffled_tensor

parser = argparse.ArgumentParser(description="extract index")

parser.add_argument('dataset')
parser.add_argument('-cep', '--content-encoder-path', default='models/content_encoder.pt')
parser.add_argument('-size', default=1024, type=int)
parser.add_argument('--stride', default=4, type=int)
parser.add_argument('-o', '--output', default='models/index.pt')
parser.add_argument('-d', '--device', default='cpu')

args = parser.parse_args()

device = torch.device(args.device) # use cpu because content encoder is lightweight.
CE = ContentEncoder().to(device).eval()
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))

features = []
total_length = 0

ds = WaveFileDirectory([args.dataset], length=16000)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

print("Extracting...")
for i, wave in enumerate(dl):
    feat = CE.encode(wave.to(device)).cpu()[:, :, ::args.stride]
    total_length += feat.shape[2]
    features.append(feat)
    if total_length > args.size:
        break

features = torch.cat(features, dim=2)
idx = shuffle(features, dim=2)[:, :, :args.size]
print(f"Extracted {idx.shape[2]} vectors.")

print("Saving...")
torch.save(idx, args.output)

print("Complete.")
