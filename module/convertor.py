import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .content_encoder import ContentEncoder
from .pitch_estimator import PitchEstimator
from .speaker_embedding import SpeakerEmbedding
from .decoder import Decoder
from .common import energy, compute_f0, oscillate_harmonics, instance_norm, incremental_instance_norm


# for realtime inferencing
class Convertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = ContentEncoder().eval()
        self.pitch_estimator = PitchEstimator().eval()
        self.speaker_embedding = SpeakerEmbedding().eval()
        self.decoder = Decoder().eval()
        self.frame_size = self.decoder.frame_size
        self.num_harmonics = self.decoder.num_harmonics
        self.sample_rate = self.decoder.sample_rate

    def load(self, path='./models', device='cpu'):
        self.pitch_estimator.load_state_dict(torch.load(os.path.join(path, 'pitch_estimator.pt'), map_location=device))
        self.content_encoder.load_state_dict(torch.load(os.path.join(path, 'content_encoder.pt'), map_location=device))
        self.speaker_embedding.load_state_dict(torch.load(os.path.join(path, 'speaker_embedding.pt'), map_location=device))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pt'), map_location=device))

    def encode_target(self, wave, stride=4):
        tgt = self.content_encoder.encode(wave)
        return tgt[:, :, ::stride]

    # convert single waveform without buffering
    @torch.inference_mode()
    def convert(self, wave, spk, pitch_shift=0, pitch_estimation_algorithm='default'):
        z = self.content_encoder.encode(wave)
        z = instance_norm(z)

        e = energy(wave)
        if pitch_estimation_algorithm != 'default':
            p = compute_f0(wave, algorithm=pitch_estimation_algorithm)
        else:
            p = self.pitch_estimator.estimate(wave)
        scale = 12 * torch.log2(p / 440)
        scale += pitch_shift
        p = 440 * 2 ** (scale / 12)
        return self.decoder.synthesize(z, p, e, spk)

    # initialize buffer for realtime inferencing
    @torch.inference_mode()
    def init_buffer(self, buffer_size, device='cpu'):
        audio_buffer = torch.zeros(1, buffer_size, device=device)
        phase_buffer = torch.zeros(1, self.num_harmonics + 1, 1, device=device)
        z_channels = self.content_encoder.output_channels
        iin_buffer = None
        return audio_buffer, phase_buffer, iin_buffer
    
    # convert voice with buffer for realtime inferencing
    @torch.inference_mode()
    def convert_rt(self, chunk, buffer, spk, pitch_shift, k=4, alpha=0, pitch_estimation='default'):
        N = chunk.shape[0]
        device = chunk.device
        k = int(k)

        # extpand buffer variables
        audio_buffer, phase_buffer, iin_buffer = buffer

        # buffer size and chunk size
        buffer_size = audio_buffer.shape[1]
        chunk_size = chunk.shape[1]

        # concateante audio buffer and chunk
        x = torch.cat([audio_buffer, chunk], dim=1)
        waveform_length = x.shape[1]

        # encode content, estimate energy, estimate pitch
        z = self.content_encoder.encode(x)
        
        # incremental instance normalization
        z, iin_buffer = incremental_instance_norm(z, iin_buffer)

        if pitch_estimation == 'default':
            p = self.pitch_estimator.estimate(x)
        else:
            p = compute_f0(x, algorithm=pitch_estimation)
        e = energy(x, self.frame_size)

        # pitch shift
        scale = 12 * torch.log2(p / 440)
        scale += pitch_shift
        p = 440 * 2 ** (scale / 12)

        # oscillate harmonics and noise
        harmonics, phase_out = oscillate_harmonics(
                p,
                phase_buffer,
                self.frame_size,
                self.sample_rate,
                self.num_harmonics,
                begin_point=buffer_size-1)

        L = harmonics.shape[2]
        N = harmonics.shape[0]
        noise = torch.randn(N, 1, L, device=device)
        src = torch.cat([harmonics, noise], dim=1)

        # calculate next phase buffer
        new_phase_buffer = phase_out[:, :, -1].unsqueeze(2)
        
        # synthesize new voice
        y = self.decoder(z, e, spk, src)

        # return new voice and shift left
        left_shift = self.frame_size * 3
        audio_out = y[:, buffer_size-left_shift:-left_shift]
        new_audio_buffer = x[:, -buffer_size:]

        return audio_out, (new_audio_buffer, new_phase_buffer, iin_buffer)
