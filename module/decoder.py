import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DCC, oscillate_harmonics


class FiLM(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.to_mu = nn.Conv1d(cond_channels, channels, 1)
        self.to_sigma = nn.Conv1d(cond_channels, channels, 1)

    def forward(self, x, c):
        mu = self.to_mu(c)
        sigma = self.to_sigma(c)
        x = x * sigma + mu
        return x


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4):
        super().__init__()
        self.factor = factor

        self.down_res = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = DCC(input_channels, input_channels, 3, 1)
        self.c2 = DCC(input_channels, input_channels, 3, 2)
        self.c3 = DCC(input_channels, output_channels, 3, 4)
        self.pool = nn.AvgPool1d(factor)

    def forward(self, x):
        res = self.down_res(x)
        x = F.gelu(x)
        x = self.c1(x)
        x = F.gelu(x)
        x = self.c2(x)
        x = F.gelu(x)
        x = self.c3(x)
        x = x + res
        x = self.pool(x)
        return x


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels, cond_channels, factor=4):
        super().__init__()
        self.factor = factor
        
        self.film1 = FiLM(input_channels, cond_channels)
        self.film2 = FiLM(input_channels, cond_channels)
        self.c1 = DCC(input_channels, input_channels, 3, 1)
        self.c2 = DCC(input_channels, input_channels, 3, 3)
        self.c3 = DCC(input_channels, input_channels, 3, 9)
        self.c4 = DCC(input_channels, input_channels, 3, 27)
        self.c5 = DCC(input_channels, output_channels, 3, 1)

    def forward(self, x, c):
        x = F.interpolate(x, scale_factor=self.factor, mode='linear')
        c = F.interpolate(c, scale_factor=self.factor, mode='linear')
        res = x
        x = F.gelu(x)
        x = self.c1(x)
        x = F.gelu(x)
        x = self.c2(x)
        x = self.film1(x, c)
        x = x + res
        res = x
        x = F.gelu(x)
        x = self.c3(x)
        x = F.gelu(x)
        x = self.c4(x)
        x = self.film2(x, c)
        x = x + res
        x = self.c5(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 channels=[256, 128, 64, 32],
                 factors=[4, 4, 5, 6],
                 cond_channels=[256, 128, 64, 32],
                 num_harmonics=0,
                 content_channels=256,
                 spk_dim=256,
                 sample_rate=24000,
                 frame_size=480,
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.content_channels = content_channels
        self.spk_dim = spk_dim

        # content input and energy input
        self.content_input = nn.Conv1d(content_channels, channels[0], 1)
        self.energy_input = nn.Conv1d(1, cond_channels[0], 1)
        self.speaker_input = nn.Conv1d(spk_dim, cond_channels[0], 1)
        self.film = FiLM(channels[0], cond_channels[0])

        # initialize downsample layers
        self.down_input = nn.Conv1d(num_harmonics + 2, cond_channels[-1], 1)
        self.downs = nn.ModuleList([])
        cond = list(reversed(cond_channels))
        cond_next = cond[1:] + [cond[-1]]
        for c, c_n, f in zip(cond, cond_next, reversed(factors)):
            self.downs.append(
                    Downsample(c, c_n, f))

        # initialize upsample layers
        self.ups = nn.ModuleList([])
        up = channels
        up_next = channels[1:] + [channels[-1]]
        for u, u_n, c_n, f in zip(up, up_next, reversed(cond_next), factors):
            self.ups.append(Upsample(u, u_n, c_n, f))
        # output layer
        self.output_layer = DCC(channels[-1], 1, 3, 1)

    def generate_source(self, p):
        L = p.shape[2] * self.frame_size
        N = p.shape[0]
        device = p.device

        # generate harmonics and noises
        harmonics, _ = oscillate_harmonics(p, 0, self.frame_size, self.sample_rate, self.num_harmonics)
        noise = torch.randn(N, 1, L, device=device)
        source_signals = torch.cat([harmonics, noise], dim=1)
        return source_signals

    def forward(self, x, e, spk, source_signals):
        # prenet
        c = self.speaker_input(spk) + self.energy_input(e)
        x = self.content_input(x)
        x = F.gelu(x)
        x = self.film(x, c)

        # downsamples
        skips = []

        # downsamples
        s = self.down_input(source_signals)
        for d in self.downs:
            s = d(s)
            skips.append(s)

        # upsamples
        for u, s in zip(self.ups, reversed(skips)):
            x = u(x, s)

        x = self.output_layer(x)
        x = x.squeeze(1)
        return x

    def synthesize(self, x, p, e, spk):
        source_signals = self.generate_source(p)
        out = self.forward(x, e, spk, source_signals)
        return out
