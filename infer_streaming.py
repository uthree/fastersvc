import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample, gain

import numpy as np
import pyaudio

from module.convertor import Convertor
from module.common import energy

FRAME_SIZE=480
INTERNAL_SR=48000


# Oscillate harmonic signal for realtime inferencing
#
# Inputs ---
# f0: [BatchSize, 1, Frames]
# phase: scaler or [BatchSize, NumHarmonics, 1]
#
# Outputs ---
# (signals, phase)
# signals: [BatchSize, NumHarmonics, Length]
# phase: [BatchSize, NumHarmonics Length]
#
# phase's range is 0 to 1, multiply 2 * pi if you need radians
# length = Frames * frame_size
def oscillate_harmonics(f0,
                        phase=0,
                        frame_size=480,
                        sample_rate=48000,
                        num_harmonics=0,
                        begin_point=0):
    N = f0.shape[0]
    Nh = num_harmonics + 1
    Lf = f0.shape[2]
    Lw = Lf * frame_size

    device = f0.device

    # generate frequency of harmonics
    mul = (torch.arange(Nh, device=device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lf)
    fs = f0 * mul

    # change length to wave's
    fs = F.interpolate(fs, Lw)

    # generate harmonics
    I = torch.cumsum(fs / sample_rate, dim=2) # numerical integration
    I = I - I[:, :, begin_point].unsqueeze(2)
    phi = (I + phase) % 1 # new phase
    theta = 2 * math.pi * phi # convert to radians
    harmonics = torch.sin(theta)

    return harmonics, phi

# the function of generating sinewaves and noise for realtime inference
def generate_source(f0, phase, begin_point):
    L = f0.shape[2] * FRAME_SIZE
    N = f0.shape[0]
    device = f0.device

    # generate noise 
    noises = torch.rand(N, 1, L, device=device)

    # generate harmonics
    sines, phase_out = oscillate_harmonics(f0,
                                           phase,
                                           FRAME_SIZE,
                                           INTERNAL_SR,
                                           phase.shape[1] - 1,
                                           begin_point)
    source_signals = torch.cat([sines, noises], dim=1)
    return source_signals, phase_out


def init_buffer(buffer_size, num_harmonics, device='cpu'):
    audio_buffer = torch.zeros(1, buffer_size, device=device)
    phase_buffer = torch.zeros(1, num_harmonics + 1, 1, device=device)
    return audio_buffer, phase_buffer


@torch.inference_mode()
@torch.no_grad()
def convert_rt(convertor,
               chunk,
               buffer,
               spk,
               pitch_shift,
               sample_rate=44100):
    # extract buffer variables
    audio_buffer, phase_buffer = buffer

    # buffer size
    buffer_size = audio_buffer.shape[1] # same to begin_point of oscillator

    # concatenate audio buffer and chunk
    con_audio = torch.cat([audio_buffer, chunk], dim=1)
        
    # encode content, estimate energy, estimate pitch
    z = convertor.content_encoder.encode(con_audio)
    p = convertor.pitch_estimator.estimate(con_audio)
    e = energy(con_audio, FRAME_SIZE)

    # pitch shift
    scale = 12 * torch.log2(p / 440) - 9
    scale += pitch_shift
    p = 440 * 2 ** ((scale + 9) / 12)

    # oscillate harmonics and noise
    src, phase_out = generate_source(p, phase_buffer, buffer_size)

    # get new phase buffer
    new_phase_buffer = phase_out[:, :, -1].unsqueeze(2)

    # synthesize voice
    y = convertor.decoder(z, p, e, spk, src)

    audio_out = y[:, buffer_size:]
    new_audio_buffer = y[:, -buffer_size:]

    return audio_out, (new_audio_buffer, new_phase_buffer)


parser = argparse.ArgumentParser(description="realtime inference")

parser.add_argument('-i', '--input', default=0, type=int)
parser.add_argument('-o', '--output', default=0, type=int)
parser.add_argument('-l', '--loopback', default=-1, type=int)
parser.add_argument('-p', '--pitch-shift', default=0, type=float)
parser.add_argument('-m', '--models', default='./models/')
parser.add_argument('-t', '--target', default='target.wav')
parser.add_argument('-no-spk', default=False, type=bool)
parser.add_argument('-c', '--chunk', default=3840, type=int)
parser.add_argument('-b', '--buffer', default=2, type=int)
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-sr', '--sample-rate', default=48000, type=int)
parser.add_argument('-ig', '--input-gain', default=0, type=float)
parser.add_argument('-og', '--output-gain', default=0, type=float)

args = parser.parse_args()

device = torch.device(args.device)

convertor = Convertor()
convertor.load(args.models)
convertor.to(device)

audio = pyaudio.PyAudio()

if args.no_spk:
    spk = torch.zeros(1, 256, 1, device=device)
else:
    print("Loading target...")
    wf, sr = torchaudio.load(args.target)
    wf = wf.to(device)
    wf = resample(wf, sr, 48000)
    wf = wf[:1]
    print("Encoding target...")
    spk = convertor.encode_speaker(wf)

stream_input = audio.open(
        format=pyaudio.paInt16,
        rate=args.sample_rate,
        channels=1,
        input_device_index=args.input,
        input=True)
stream_output = audio.open(
        format=pyaudio.paInt16,
        rate=args.sample_rate, 
        channels=1,
        output_device_index=args.output,
        output=True)
stream_loopback = audio.open(
        format=pyaudio.paInt16,
        rate=args.sample_rate, 
        channels=1,
        output_device_index=args.loopback,
        output=True) if args.loopback != -1 else None

BUFFER_SIZE = args.buffer * args.chunk
CHUNK_SIZE = args.chunk
N_HARM = convertor.decoder.num_harmonics

# initialize buffer
buffer = init_buffer(BUFFER_SIZE, N_HARM, device)

# inference loop
print("Converting voice, Ctrl+C to stop conversion")
while True:
    chunk = stream_input.read(CHUNK_SIZE)
    chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    chunk = torch.from_numpy(chunk).to(device)
    chunk = chunk.unsqueeze(0) / 32768
    
    chunk = gain(chunk, args.input_gain)
    chunk, buffer = convert_rt(convertor, chunk, buffer, spk, args.pitch_shift)
    chunk = gain(chunk, args.output_gain)

    chunk = chunk.cpu().numpy() * 32768
    chunk = chunk.astype(np.int16).tobytes()
    stream_output.write(chunk)
    if stream_loopback is not None:
        stream_loopback.write(chunk)
