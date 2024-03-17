import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .content_encoder import ContentEncoder
from .pitch_estimator import PitchEstimator
from .decoder import Decoder
from .common import energy, match_features, compute_f0, oscillate_harmonics


# for realtime inferencing
class Convertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = ContentEncoder().eval()
        self.pitch_estimator = PitchEstimator().eval()
        self.decoder = Decoder().eval()
        self.frame_size = self.decoder.frame_size
        self.num_harmonics = self.decoder.num_harmonics
        self.sample_rate = self.decoder.sample_rate

    def load(self, path='./models', device='cpu'):
        self.pitch_estimator.load_state_dict(torch.load(os.path.join(path, 'pitch_estimator.pt'), map_location=device))
        self.content_encoder.load_state_dict(torch.load(os.path.join(path, 'content_encoder.pt'), map_location=device))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pt'), map_location=device))

    def encode_target(self, wave, downsample_factor=4):
        tgt = self.content_encoder.encode(wave)
        tgt = F.avg_pool1d(tgt, downsample_factor)
        return tgt

    # convert single waveform without buffering
    @torch.inference_mode()
    def convert(self, wave, tgt, pitch_shift=0, k=4, alpha=0, pitch_estimation_algorithm='default'):
        z = self.content_encoder.encode(wave)

        z = match_features(z, tgt, k, alpha)
        l = energy(wave)
        if pitch_estimation_algorithm != 'default':
            p = compute_f0(wave, algorithm=pitch_estimation_algorithm)
        else:
            p = self.pitch_estimator.estimate(wave)
        scale = 12 * torch.log2(p / 440)
        scale += pitch_shift
        p = 440 * 2 ** (scale / 12)
        return self.decoder.synthesize(z, p, l)

    # initialize buffer for realtime inferencing
    @torch.inference_mode()
    def init_buffer(self, buffer_size, device='cpu'):
        input_buffer = torch.zeros(1, buffer_size, device=device)
        output_buffer = torch.zeros(1, buffer_size, device=device)
        return input_buffer, output_buffer
    
    # convert voice with buffer for realtime inferencing
    @torch.inference_mode()
    def convert_rt(self, chunk, buffer, tgt, pitch_shift, k=4, alpha=0, pitch_estimation='default'):
        N = chunk.shape[0]
        device = chunk.device
        k = int(k)

        # extpand buffer variables
        input_buffer, output_buffer = buffer

        # buffer size and chunk size
        buffer_size = input_buffer.shape[1]
        chunk_size = chunk.shape[1]

        # concateante audio buffer and chunk
        x = torch.cat([input_buffer, chunk], dim=1)
        waveform_length = x.shape[1]

        # encode content, estimate energy, estimate pitch
        z = self.content_encoder.encode(x)
        if pitch_estimation == 'default':
            p = self.pitch_estimator.estimate(x)
        else:
            p = compute_f0(x, algorithm=pitch_estimation)
        e = energy(x, self.frame_size)

        # convert style
        z = match_features(z, tgt, k, alpha)

        # pitch shift
        scale = 12 * torch.log2(p / 440)
        scale += pitch_shift
        p = 440 * 2 ** (scale / 12)

        # oscillate harmonics and noise
        harmonics, _ = oscillate_harmonics(
                p,
                0,
                self.frame_size,
                self.sample_rate,
                self.num_harmonics)

        noise = torch.randn(N, 1, waveform_length, device=device)

        src = torch.cat([harmonics, noise], dim=1)

        # synthesize new voice
        y = self.decoder(z, p, e, src)

        # cross fade (linear interpolation)
        y_hat = torch.cat([output_buffer, torch.zeros(N, chunk_size, device=device)], dim=1)
        alpha = torch.cat([
            torch.zeros(buffer_size - chunk_size),
            torch.linspace(0, 1.0, chunk_size),
            torch.ones(chunk_size),
            ]).to(device).unsqueeze(0)
        alpha = alpha.expand(N, alpha.shape[1])
        new_output_buffer = y_hat * (1-alpha) + y * alpha

        left_shift = chunk_size
        out_signal = new_output_buffer[:, -chunk_size-left_shift:-left_shift]

        new_output_buffer = y[:, -buffer_size:]
        new_input_buffer = x[:, -buffer_size:]
        return out_signal, (new_input_buffer, new_output_buffer)
