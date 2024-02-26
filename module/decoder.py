import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DCC, oscillate_harmonics


class FiLM(nn.Module):
    def __init__(self, channels, cond_channels, weight_norm):
        super().__init__()
        self.to_mu = DCC(cond_channels, channels, 1, weight_norm=weight_norm)
        self.to_sigma = DCC(cond_channels, channels, 1, weight_norm=weight_norm)

    def forward(self, x, c):
        mu = self.to_mu(c)
        sigma = self.to_sigma(c)
        x = x * sigma + mu
        return x


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4, weignt_norm=True, causal=True):
        super().__init__()
        self.factor = factor

        self.down_res = DCC(input_channels, output_channels, 1, 1, 1, weignt_norm, causal)
        self.c1 = DCC(input_channels, input_channels, 3, 1, 1, weignt_norm, causal)
        self.c2 = DCC(input_channels, input_channels, 3, 2, 1, weignt_norm, causal)
        self.c3 = DCC(input_channels, output_channels, 3, 4, 1, weignt_norm, causal)
        self.pool = nn.AvgPool1d(factor)

    def forward(self, x):
        x = self.pool(x)
        res = self.down_res(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c3(x)
        x = x + res
        return x


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilations, weight_norm, causal):
        super().__init__()
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])
        for d in dilations:
            self.convs1.append(
                    DCC(channels, channels, kernel_size, d, 1, weight_norm, causal))
            self.convs1.append(
                    DCC(channels, channels, kernel_size, 1, 1, weight_norm, causal)) 

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            res = x
            x = F.leaky_relu(x, 0.1)
            x = self.c1(x)
            x = F.leaky_relu(x, 0.1)
            x = self.c2(x)
            x = x + res
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size, dilations, causal, weight_norm):
        super().__init__()
        self.convs = nn.ModuleList([])
        for d in dilations:
            self.convs.append(DCC(channels, channels, kernel_size, d, 1, weight_norm, causal))

    def forward(self, x):
        for c in self.convs:
            res = x
            x = F.leaky_relu(x, 0.1)
            x = c(x)
            x = x + res
        return x


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels, cond_channels, factor, kernel_sizes, dilations, weight_norm, causal, resblock_type):
        super().__init__()
        self.factor = factor
        self.film = FiLM(input_channels, cond_channels, weight_norm)
        self.num_kernels = len(dilations)
        self.res_blocks = nn.ModuleList([])
        if resblock_type == '1':
            resblock = ResBlock1
        elif resblock_type == '2':
            resblock = ResBlock2
        for k, ds in zip(kernel_sizes, dilations):
            self.res_blocks.append(
                    resblock(input_channels, k, ds, causal, weight_norm))
        self.out_conv = DCC(input_channels, output_channels, 3, 1, 1, weight_norm, causal)

    def forward(self, x, c):
        x = self.film(x, c)
        x = F.interpolate(x, scale_factor=self.factor, mode='linear')
        xs = None
        for b in self.res_blocks:
            if xs is None:
                xs = b(x)
            else:
                xs += b(x)
        x = xs / self.num_kernels
        x = self.out_conv(x)
        return x


class AcousticModel(nn.Module):
    def __init__(self, content_channels, channels, spk_dim, num_layers=6, kernel_size=3, causal=True, weight_norm=True):
        super().__init__()
        self.content_in = DCC(content_channels, channels, kernel_size, 1, 1, weight_norm, causal)
        self.energy_in = DCC(1, channels, 1, 1, 1, weight_norm, causal)
        self.spk_in = DCC(spk_dim, channels, 1, 1, 1, weight_norm, causal)
        self.film = FiLM(channels, channels, weight_norm)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                    DCC(channels, channels, kernel_size, 1, 1, weight_norm, causal))

    def forward(self, x, e, spk):
        x = self.content_in(x)
        c = self.energy_in(e) + self.spk_in(spk)
        x = self.film(x, c)
        for l in self.layers:
            x = F.leaky_relu(x, 0.1)
            x = l(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 resblock_type='2',
                 channels=[256, 128, 64, 32],
                 kernel_sizes=[3, 5, 7],
                 dilations=[[1, 2], [2, 6], [3, 12]],
                 factors=[4, 4, 5, 6],
                 cond_channels=[256, 128, 64, 32],
                 num_harmonics=0,
                 content_channels=256,
                 spk_dim=256,
                 sample_rate=24000,
                 causal=True,
                 weight_norm=True,
                 frame_size=480,
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.content_channels = content_channels
        self.spk_dim = spk_dim

        # content input and energy to hidden features
        self.acoustic_model = AcousticModel(content_channels, channels[0], spk_dim, causal=causal, weight_norm=weight_norm)

        # initialize downsample layers
        self.down_input = DCC(num_harmonics + 2, cond_channels[-1], 1, 1, 1, weight_norm, causal)
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
                    Upsample(u, u_n, c_n, f, kernel_sizes, dilations, weight_norm, causal, resblock_type))
        # output layer
        self.output_layer = DCC(channels[-1], 1, 7, 1, 1, weight_norm, causal)

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
        # acoustic model
        x = self.acoustic_model(x, e, spk)

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
