import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .content_encoder import ContentEncoder
from .speaker_encoder import SpeakerEncoder
from .pitch_estimator import PitchEstimator
from .decoder import Decoder
from .common import energy


# for inferencing
class Convertor(nn.Module):
    def __init__(self, frame_size=480):
        super().__init__()
        self.content_encoder = ContentEncoder().eval()
        self.pitch_estimator = PitchEstimator().eval()
        self.speaker_encoder = SpeakerEncoder().eval()
        self.decoder = Decoder().eval()
        self.frame_size = frame_size

    def load(self, path='./models', device='cpu'):
        self.pitch_estimator.load_state_dict(torch.load(os.path.join(path, 'pitch_estimator.pt'), map_location=device))
        self.speaker_encoder.load_state_dict(torch.load(os.path.join(path, 'speaker_encoder.pt'), map_location=device))
        self.content_encoder.load_state_dict(torch.load(os.path.join(path, 'content_encoder.pt'), map_location=device))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pt'), map_location=device))

    def encode_speaker(self, wave):
        return self.speaker_encoder.encode(wave)
    
    def convert(self, wave, spk, pitch_shift=0, alpha=0.2):
        # Conversion
        z = self.content_encoder.encode_infer(wave, alpha)
        l = energy(wave)
        p = self.pitch_estimator.estimate(wave)
        scale = 12 * torch.log2(p / 440) - 9
        scale += pitch_shift
        p = 440 * 2 ** ((scale + 9) / 12)
        return self.decoder.synthesize(z, p, l, spk)
