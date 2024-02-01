import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DCC


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
                        frame_size=320,
                        sample_rate=16000,
                        num_harmonics=0):
    N = f0.shape[0]
    Nh = num_harmonics + 1
    Lf = f0.shape[2]
    Lw = Lf * frame_size

    device = f0.device

    # generate frequency of harmonics
    mul = (torch.arange(Nh, device=device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lf)
    fs = f0 * mul

    # change length to wave's
    fs = F.interpolate(fs, Lw, mode='linear')

    # generate harmonics
    I = torch.cumsum(fs / sample_rate, dim=2) # numerical integration
    phi = (I + phase) % 1 # new phase
    theta = 2 * math.pi * phi # convert to radians
    harmonics = torch.sin(theta)

    return harmonics, phi


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


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4, negative_slope=0.1):
        super().__init__()
        self.negative_slope = negative_slope

        if factor % 2 == 0:
            pad_l = factor // 2
            pad_r = factor // 2
        else:
            pad_l = factor // 2 + 1
            pad_r = factor // 2
        self.pad_l = pad_l
        self.pad_r = pad_r

        self.down = nn.AvgPool1d(factor*2, factor)
        self.down_res = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = DCC(input_channels, input_channels, 5, 1)
        self.c2 = DCC(input_channels, input_channels, 5, 2)
        self.c3 = DCC(input_channels, output_channels, 5, 4)

    def forward(self, x):
        x = F.pad(x, (self.pad_l, self.pad_r), mode='replicate')
        x = self.down(x)
        res = self.down_res(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.c1(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.c2(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.c3(x)
        return x + res


class ResidualUnit1(nn.Module):
    def __init__(self, channels, kernel_size=5, dilations=[1, 3, 9], negative_slope=0.1):
        super().__init__()
        self.negative_slope = negative_slope
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])
        for d in dilations:
            self.convs1.append(DCC(channels, channels, kernel_size, d))
            self.convs2.append(DCC(channels, channels, kernel_size, d))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            res = x
            x = c1(x)
            x = F.leaky_relu(x, self.negative_slope)
            x = c2(x)
            x = F.leaky_relu(x, self.negative_slope)
            x = x + res
        return x


class Upsample(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 cond_channels,
                 factor=4,
                 negative_slope=0.1,
                 dilations=[[1, 3, 9]],
                 kernel_sizes=[5],
                 residual_unit_type='1'
                 ):
        super().__init__()
        self.negative_slope = negative_slope
        self.up = nn.Upsample(scale_factor=factor, mode='linear')
        self.film = FiLM(input_channels, cond_channels)
        if residual_unit_type == '1':
            unit_constructor = ResidualUnit1

        self.res_units = nn.ModuleList([])
        for ds, k in zip(dilations, kernel_sizes):
            self.res_units.append(unit_constructor(input_channels, k, ds))
        self.output_layer = DCC(input_channels, output_channels, 3, 1)

    def forward(self, x, c):
        x = self.film(x, c)
        x = self.up(x)
        xs = None
        for res_unit in self.res_units:
            if xs is None:
                xs = res_unit(x)
            else:
                xs += res_unit(x)
        x = self.output_layer(xs)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels, cond_channels, dilations=[[1, 3, 9]], kernel_sizes=[5], residual_unit_type='1'):
        super().__init__()
        self.film = FiLM(channels, cond_channels)
        self.res_units = nn.ModuleList([])
        if residual_unit_type == '1':
            unit_constructor = ResidualUnit1
        for k, ds in zip(kernel_sizes, dilations):
            self.res_units.append(
                    unit_constructor(channels, k, ds))
        self.output_layer = DCC(channels, channels, 3, 1)

    def forward(self, x, c):
        x = self.film(x, c)
        xs = None
        for res_unit in self.res_units:
            if xs is None:
                xs = res_unit(x)
            else:
                xs += res_unit(x)
        x = self.output_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 channels=[320, 160, 80, 40],
                 factors=[4, 4, 4, 5],
                 cond_channels=[320, 160, 80, 40],
                 num_harmonics=0, # F0 sinewave only
                 content_channels=512,
                 sample_rate=16000,
                 frame_size=320,
                 dilations=[[1, 3, 9]],
                 kernel_sizes=[5],
                 residual_unit_type='1'
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.content_channels = content_channels

        self.e2v = Energy2Vec(cond_channels[0])
        self.p2v = Pitch2Vec(cond_channels[0])
        self.content_in = nn.Conv1d(content_channels, channels[0], 1)
        self.mid_block = MidBlock(channels[0], cond_channels[0], dilations, kernel_sizes, residual_unit_type)

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
                    Upsample(u,
                             u_n,
                             c_n,
                             f,
                             dilations=dilations,
                             kernel_sizes=kernel_sizes,
                             residual_unit_type=residual_unit_type))
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

    def forward(self, x, p, e, source_signals):
        # downsamples
        skips = []
        sines = self.down_input(source_signals)
        for d in self.downs:
            sines = d(sines)
            skips.append(sines)
        
        # mid block
        c = self.e2v(e) + self.p2v(p)
        x = self.content_in(x)
        x = self.mid_block(x, c)

        # upsamples
        for u, s in zip(self.ups, reversed(skips)):
            x = u(x, s)

        x = self.output_layer(x)
        x = x.squeeze(1)
        return x

    def synthesize(self, x, p, e):
        source_signals = self.generate_source(p)
        out = self.forward(x, p, e, source_signals)
        return out
