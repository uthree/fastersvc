import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .content_encoder import ContentEncoder
from .speaker_encoder import SpeakerEncoder
from .pitch_estimator import PitchEstimator
from .decoder import Decoder
from .common import energy


class Convertor(nn.Module):
    def __init__(self, frame_size=480):
        super().__init__()
        self.content_encoder = ContentEncoder()
        self.pitch_estimator = PitchEstimator()
        self.speaker_encoder = SpeakerEncoder()
        self.decoder = Decoder()
        self.frame_size = frame_size

    def load(self, path='./models', device='cpu'):
        self.pitch_estimator.load_state_dict(torch.load(os.path.join(path, 'pitch_estimator.pt'), map_location=device))
        self.speaker_encoder.load_state_dict(torch.load(os.path.join(path, 'speaker_encoder.pt'), map_location=device))
        self.content_encoder.load_state_dict(torch.load(os.path.join(path, 'content_encoder.pt'), map_location=device))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pt'), map_location=device))

    def encode_speaker(self, wave):
        return self.speaker_encoder.encode(wave)

    def convert(self, wave, spk, pitch_shift=0):
        # Padding
        N = wave.shape[0]
        pad_len = self.frame_size - (wave.shape[1] % self.frame_size)
        pad = torch.zeros(N, pad_len, device=wave.device)
        wave = torch.cat([wave, pad], dim=1)
        
        # Conversion
        z = self.content_encoder.encode(wave)
        l = energy(wave)
        p = self.pitch_estimator.estimate(wave)
        scale = 12 * torch.log2(p / 440) - 9
        scale += pitch_shift
        p = 440 * 2 ** ((scale + 9) / 12)
        return self.decoder(z, p, l, spk)
