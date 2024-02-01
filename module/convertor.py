import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .content_encoder import ContentEncoder
from .pitch_estimator import PitchEstimator
from .decoder import Decoder
from .common import energy, match_features
from .adain import encode_style, apply_style
from .dataset import compute_f0


# for inferencing
class Convertor(nn.Module):
    def __init__(self, frame_size=320):
        super().__init__()
        self.content_encoder = ContentEncoder().eval()
        self.pitch_estimator = PitchEstimator().eval()
        self.decoder = Decoder().eval()
        self.frame_size = frame_size

    def load(self, path='./models', device='cpu'):
        self.pitch_estimator.load_state_dict(torch.load(os.path.join(path, 'pitch_estimator.pt'), map_location=device))
        self.content_encoder.load_state_dict(torch.load(os.path.join(path, 'content_encoder.pt'), map_location=device))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pt'), map_location=device))

    def encode_target(self, wave, stride=4):
        tgt = self.content_encoder.encode(wave)
        return tgt[:, :, ::stride]

    def convert(self, wave, tgt, pitch_shift=0, k=4, alpha=0, adain=False, pitch_estimation_algorithm='default'):
        # Conversion
        z = self.content_encoder.encode(wave)
        if adain:
            style = encode_style(tgt)
            z = apply_style(z, style)
        z = match_features(z, tgt, k, alpha)
        l = energy(wave)
        if pitch_estimation_algorithm != 'default':
            p = compute_f0(wave, algorithm=pitch_estimation_algorithm)
        else:
            p = self.pitch_estimator.estimate(wave)
        scale = 12 * torch.log2(p / 440) - 9
        scale += pitch_shift
        p = 440 * 2 ** ((scale + 9) / 12)
        return self.decoder.synthesize(z, p, l)
