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

    def forward(self, x):
        x = F.interpolate(x, scale_factor=1/self.factor, mode='linear')
        res = self.down_res(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c3(x)
        return x + res


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels, cond_channels, factor=4):
        super().__init__()
        self.factor = factor
        
        self.film1 = FiLM(input_channels, cond_channels)
        self.film2 = FiLM(input_channels, cond_channels)
        self.film3 = FiLM(input_channels, cond_channels)
        self.c1 = DCC(input_channels, input_channels, 3, 1)
        self.c2 = DCC(input_channels, input_channels, 3, 3)
        self.c3 = DCC(input_channels, input_channels, 3, 9)
        self.c4 = DCC(input_channels, input_channels, 3, 27)
        self.c5 = DCC(input_channels, output_channels, 3, 1)

    def forward(self, x, c):
        x = F.interpolate(x, scale_factor=self.factor, mode='linear')
        c = F.interpolate(c, scale_factor=self.factor, mode='linear')
        res = x
        x = F.leaky_relu(x, 0.1)
        x = self.c1(x)
        x = self.film1(x, c)
        x = F.leaky_relu(x, 0.1)
        x = self.c2(x)
        x = x + res
        res = x
        x = F.leaky_relu(x, 0.1)
        x = self.c3(x)
        x = self.film2(x, c)
        x = F.leaky_relu(x, 0.1)
        x = self.c4(x)
        x = self.film3(x, c)
        x = x + res
        x = self.c5(x)
        return x


class PreNet(nn.Module):
    def __init__(self,
                 content_channels=256,
                 internal_channels=256,
                 output_channels=256,
                 cond_channels=256,
                 spk_dim=256,
                 num_harmonics=15,
                 frame_size=480,
                 kernel_size=3):
        super().__init__()
        self.frame_size = frame_size
        self.num_harmoncis = num_harmonics
        
        self.spk_in = nn.Conv1d(spk_dim, cond_channels, 1)
        self.energy_in = nn.Conv1d(1, cond_channels, 1)
        self.c1 = DCC(content_channels, internal_channels, 3)
        self.film1 = FiLM(internal_channels, cond_channels)
        self.c2 = DCC(internal_channels, internal_channels, 3)
        self.film2 = FiLM(internal_channels, cond_channels)
        self.output_layer = DCC(internal_channels, internal_channels, 3)
        self.to_harmonic_amps = DCC(internal_channels, num_harmonics+1, 3)

    def forward(self, x, e, spk):
        c = self.energy_in(e) + self.spk_in(spk)
        x = self.c1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.film1(x, c)
        x = self.c2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.film2(x, c)
        out = self.output_layer(x)
        amps = self.to_harmonic_amps(x)
        # positive only
        amps = torch.exp(amps).clamp_max(6.0)
        amps = F.interpolate(amps, scale_factor=self.frame_size, mode='linear')
        return out, amps


class Decoder(nn.Module):
    def __init__(self,
                 channels=[256, 128, 64, 32],
                 factors=[4, 4, 5, 6],
                 cond_channels=[256, 128, 64, 32],
                 num_harmonics=15,
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

        # initialize prenet
        self.prenet = PreNet(content_channels, channels[0], channels[0],
                             cond_channels[0], spk_dim, num_harmonics, frame_size)

        # initialize downsample layers
        self.down_input = nn.Conv1d(1, cond_channels[-1], 1)
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

        # generate harmonics
        source_signals, _ = oscillate_harmonics(p, 0, self.frame_size, self.sample_rate, self.num_harmonics)
        return source_signals

    def forward(self, x, e, spk, source_signals):
        # pass prenet and estimate harmonic ampliudes
        x, amps = self.prenet(x, e, spk)

        # downsamples
        skips = []
        
        # additive sinthesizer
        s = torch.sum(source_signals * amps, dim=1, keepdim=True)

        # downsamples
        s = self.down_input(s)
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
