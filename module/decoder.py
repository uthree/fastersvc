import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DCC

LRELU_SLOPE = 0.1

# Oscillate harmonic signal
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
                        num_harmonics=1):
    N = f0.shape[0]
    Nh = num_harmonics
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
    phi = (I + phase) % 1 # new phase
    theta = 2 * math.pi * phi # convert to radians
    harmonics = torch.sin(theta)

    return harmonics, phi


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4):
        super().__init__()
        self.down = nn.AvgPool1d(factor)
        self.down_res = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = DCC(input_channels, input_channels, 3, 1)
        self.c2 = DCC(input_channels, input_channels, 3, 2)
        self.c3 = DCC(input_channels, output_channels, 3, 4)

    def forward(self, x):
        x = self.down(x)
        res = self.down_res(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c1(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c2(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c3(x)
        return x + res


class Pitch2Vec(nn.Module):
    def __init__(self, cond_channels):
        super().__init__()
        self.c1 = nn.Conv1d(1, cond_channels, 1, 1, 0)
        self.c2 = nn.Conv1d(cond_channels, cond_channels, 1, 1, 0)
        self.c1.weight.data.normal_(0, 0.5)

    def forward(self, x):
        x = self.c1(x)
        x = torch.sin(x)
        x = self.c2(x)
        return x


class Energy2Vec(nn.Module):
    def __init__(self, cond_channels):
        super().__init__()
        self.c1 = nn.Conv1d(1, cond_channels, 1, 1, 0)

    def forward(self, x):
        return self.c1(x)


class FiLM(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.to_mu = nn.Conv1d(cond_channels, channels, 1)
        self.to_sigma = nn.Conv1d(cond_channels, channels, 1)

    def forward(self, x, c):
        mu = self.to_mu(c)
        sigma = self.to_sigma(c)
        x = x * mu + sigma
        return x


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels, cond_channels, factor=4, kernel_size=7):
        super().__init__()
        self.up = nn.Upsample(scale_factor=factor, mode='linear')

        self.c1 = DCC(input_channels, input_channels, 3, 1)
        self.c2 = DCC(input_channels, input_channels, 3, 3)
        self.film1 = FiLM(input_channels, cond_channels)

        self.c3 = DCC(input_channels, input_channels, 3, 9)
        self.c4 = DCC(input_channels, input_channels, 3, 27)
        self.film2 = FiLM(input_channels, cond_channels)

        self.out_conv = nn.Conv1d(input_channels, output_channels, 1)

    def forward(self, x, c):
        x = self.up(x)
        c = self.up(c)
        res = x
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c1(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c2(x)
        x = self.film1(x, c)
        x = x + res
        res = x
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c3(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.c4(x)
        x = self.film2(x, c)
        x = x + res
        x = self.out_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 channels=[192, 96, 48, 24],
                 factors=[4, 4, 5, 6],
                 cond_channels=[192, 96, 48, 24],
                 num_harmonics=0, # F0 sinewave only
                 speaker_channels=256,
                 content_channels=32,
                 sample_rate=48000,
                 frame_size=480,
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics + 1
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        # initialize first layer
        self.up_input = nn.Conv1d(content_channels, channels[0], 1)
        self.up_film = FiLM(channels[0], speaker_channels)
        self.e2v = Energy2Vec(speaker_channels)
        self.p2v = Pitch2Vec(speaker_channels)
        
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
            self.ups.append(
                    Upsample(u, u_n, c_n, f))

        # output layer
        self.output_layer = DCC(channels[-1], 1, 3, 1)

    def generate_source(self, p):
        L = p.shape[2] * self.frame_size
        N = p.shape[0]
        device = p.device

        # generate noise
        noises = torch.rand(N, 1, L, device=device)
        
        # generate harmonics
        sines, _ = oscillate_harmonics(p, 0, self.frame_size, self.sample_rate, self.num_harmonics)
        source_signals = torch.cat([sines, noises], dim=1)
        return source_signals

    def forward(self, x, p, e, spk, source_signals):
        # downsamples
        skips = []
        sines = self.down_input(source_signals)
        for d in self.downs:
            sines = d(sines)
            skips.append(sines)

        # upsamples
        cond = spk + self.e2v(e) + self.p2v(p)
        x = self.up_film(self.up_input(x), cond)
        for u, s in zip(self.ups, reversed(skips)):
            x = u(x, s)

        x = self.output_layer(x)
        x = x.squeeze(1)
        return x

    def synthesize(self, x, p, e, spk):
        source_signals = self.generate_source(p)
        out = self.forward(x, p, e, spk, source_signals)
        return out
