import argparse
import os
from pathlib import Path
import json

import torch
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.common import compute_f0


parser = argparse.ArgumentParser(description="preprocess")

parser.add_argument('input')
parser.add_argument('-o', '--output', default='dataset_cache')
parser.add_argument('-len', '--length', default=48000, type=int)
parser.add_argument('--num-speakers', default=8192, type=int)
parser.add_argument('-m', '--max-files', default=-1, type=int)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('--speaker-infomation', default='speaker_infomation.json')

args = parser.parse_args()

device = torch.device(args.device)

input_parent = Path(args.input)
dataset_files = []

support_exts = ['mp3', 'wav', 'ogg']
for e in support_exts:
    dataset_files += list(input_parent.glob(f"**/*.{e}"))
if args.max_files != -1:
    dataset_files = dataset_files[:args.max_files]

hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
hubert = hubert.to(device)

# create output directory
output_parent = Path(args.output)
if not output_parent.exists():
    output_parent.mkdir()

parent_paths = []
counter = 0
for path in tqdm(dataset_files):
    tqdm.write(f"processing {str(path)}")
    parent_path = path.parent
    wf, sr = torchaudio.load(path)
    wf = wf.mean(dim=0, keepdim=True)
    wf = resample(wf, sr, 24000)
    # chunk
    chunks = wf.split(args.length, dim=1)
    for chunk in chunks:
        if chunk.shape[1] < args.length:
            # padding
            pad_len = args.length - chunk.shape[1]
            pad = torch.zeros(1, pad_len)
            chunk = torch.cat([chunk, pad], dim=1)

        # f0
        f0 = compute_f0(chunk)

        # extract hubert features
        hubert_input = resample(chunk, 24000, 16000).to(device).unsqueeze(1)
        hubert_features = hubert.units(hubert_input).to(device).transpose(1, 2)

        # get spekaer id
        if parent_path not in parent_paths:
            parent_paths.append(parent_path)
        spk_id = parent_paths.index(parent_path)
        spk_id = min(spk_id, args.num_speakers)

        # save
        output_path = output_parent / f"{counter}.pt"
        torch.save((chunk[0].detach().cpu(), f0[0].detach().cpu(), hubert_features[0].detach().cpu(), spk_id), output_path)
        counter += 1

# output speaker details
output_obj = {}
for i, p in enumerate(parent_paths):
    output_obj[str(i)] = str(p)

with open(args.speaker_infomation, 'w') as f:
    json.dump(output_obj, f)

print("complete!")
