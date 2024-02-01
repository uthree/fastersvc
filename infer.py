import argparse
import os
import glob

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.convertor import Convertor

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('-m', '--models', default='./models/')
parser.add_argument('-p', '--pitch-shift', default=0, type=float)
parser.add_argument('-t', '--target', default='./target.wav')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-a', '--alpha', default=0, type=float)
parser.add_argument('-idx', '--index', default='NONE')
parser.add_argument('--normalize', default=False, type=bool)
parser.add_argument('-adain', default=False, type=bool)
parser.add_argument('-pe', '--pitch-estimation', default='default', choices=['default', 'dio', 'harvest'])
parser.add_argument('-c', '--chunk', default=1280, type=int)
parser.add_argument('-b', '--buffer', default=3, type=int)

args = parser.parse_args()

device = torch.device(args.device)

convertor = Convertor()
convertor.load(args.models)
convertor.to(device)

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)


if args.index == 'NONE':
    print("Loading target...")
    wf, sr = torchaudio.load(args.target)
    wf = wf.to(device)
    wf = resample(wf, sr, 16000)
    wf = wf[:1]
    print("Encoding...")
    tgt = convertor.encode_target(wf)
else:
    print("Loading index...")
    tgt = torch.load(args.index).to(device)


paths = glob.glob(os.path.join(args.inputs, "*"))
left_shift = convertor.frame_size * 3
buffer_size = args.buffer * args.chunk
for i, path in enumerate(paths):
    print(f"Converting {path} ...")
    wf, sr = torchaudio.load(path)
    wf = resample(wf, sr, 16000)
    wf = wf.mean(dim=0, keepdim=True)
    chunks = torch.split(wf, args.chunk, dim=1)
    results = []
    buffer = convertor.init_buffer(buffer_size, device=device)
    for chunk in tqdm(chunks):
        if chunk.shape[1] < args.chunk:
            pad_len = args.chunk - chunk.shape[1]
            chunk = torch.cat([chunk, torch.zeros(1, pad_len)], dim=1)
        converted_chunk, buffer = convertor.convert_rt(
                chunk.to(device),
                buffer,
                tgt,
                args.pitch_shift,
                alpha=args.alpha,
                pitch_estimation=args.pitch_estimation,
                )
        results.append(converted_chunk.cpu())
    wf = torch.cat(results, dim=1)
    wf = wf[:, (left_shift):]
    wf = resample(wf, 16000, sr)
    file_name = f"{os.path.splitext(os.path.basename(path))[0]}"
    torchaudio.save(os.path.join(args.outputs, f"{file_name}.wav"), src=wf, sample_rate=sr)
