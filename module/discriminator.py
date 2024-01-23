import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class PeriodicDiscriminator(nn.Module):
    def __init__(self,
                 channels=32,
                 period=2,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 groups = [1, 1, 1, 1, 1],
                 max_channels=128,
                 ):
        super().__init__()
        self.input_layer = weight_norm(
                nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), padding=get_padding(kernel_size, 1)))
        self.layers = nn.Sequential()
        for i in range(num_stages):
            c = min(channels * (4 ** i), max_channels)
            c_next = min(channels * (4 ** (i+1)), max_channels)
            if i == (num_stages - 1):
                self.layers.append(
                        weight_norm(
                            nn.Conv2d(c, c, (kernel_size, 1), (stride, 1), groups=groups[i],
                                      padding=get_padding(kernel_size, 1))))
            else:
                self.layers.append(
                        weight_norm(
                            nn.Conv2d(c, c_next, (kernel_size, 1), (stride, 1), groups=groups[i],
                                      padding=get_padding(kernel_size, 1))))
                self.layers.append(
                        nn.LeakyReLU(LRELU_SLOPE))
        c = min(channels * (4 ** (num_stages-1)), max_channels)
        self.final_conv = weight_norm(
                nn.Conv2d(c, c, (5, 1), 1, padding=get_padding(5, 1)))
        self.final_relu = nn.LeakyReLU(LRELU_SLOPE)
        self.output_layer = weight_norm(
                nn.Conv2d(c, 1, (3, 1), 1, padding=get_padding(3, 1)))
        self.period = period

    def forward(self, x):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            if "Conv" in type(layer).__name__:
                feats.append(x)
        x = self.final_conv(x)
        x = self.final_relu(x)
        x = self.output_layer(x)
        return x, feats


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(self,
                 periods=[1, 3, 5, 7, 17, 23, 37],
                 groups=[1, 4, 4, 4, 4, 4],
                 channels=32,
                 kernel_size=5,
                 stride=3,
                 num_stages=5):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for p in periods:
            self.sub_discs.append(
                    PeriodicDiscriminator(channels,
                                          p,
                                          kernel_size,
                                          stride,
                                          num_stages,
                                          groups=groups))
    def logits(self, x):
        logits = []
        for sd in self.sub_discs:
            l, _ = sd(x)
            logits.append(l)
        return logits

    def feat_loss(self, fake, real):
        loss = 0
        for sd in self.sub_discs:
            _, feat_f = sd(fake)
            _, feat_r = sd(real)
            for a, b in zip(feat_f, feat_r):
                loss += (a - b).abs().mean()
        return loss


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft, channels):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.channels = channels

        self.layers = nn.ModuleList([
                weight_norm(nn.Conv2d(1, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
                weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
            ])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        window = torch.hann_window(self.n_fft, device=x.device)
        x = torch.stft(x, self.n_fft, self.hop_length, return_complex=True, window=window).abs()
        x = x.unsqueeze(1)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
        return x, feats


class MultiSTFTDiscriminator(nn.Module):
    def __init__(self, n_ffts=[1024, 2048, 4096], channels=32):
        super().__init__()
        self.sub_discs = nn.ModuleList([STFTDiscriminator(s, channels) for s in n_ffts])
    
    def logits(self, x):
        logits = []
        for sd in self.sub_discs:
            l, _ = sd(x)
            logits.append(l)
        return logits

    def feat_loss(self, fake, real):
        loss = 0
        for sd in self.sub_discs:
            _, feat_f = sd(fake)
            _, feat_r = sd(real)
            for a, b in zip(feat_f, feat_r):
                loss += (a - b).abs().mean()
        return loss


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodicDiscriminator()
        self.mrd = MultiSTFTDiscriminator()

    def logits(self, x):
        return self.mpd.logits(x) + self.mrd.logits(x)

    def feat_loss(self, x, y):
        return self.mpd.feat_loss(x, y) + self.mrd.feat_loss(x, y)

