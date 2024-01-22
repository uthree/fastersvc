import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResBlock, spectrogram


class VectorExplorer(nn.Module):
    def __init__(self, num_centroids=128, channels=256):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(1, channels, num_centroids))
        self.channels = channels

    def forward(self, source, k=4, alpha=0.0):
        return self.match(source, k, alpha)

    def match(self, source, k=4, alpha=0.0):
        reference = self.centroids.expand(
                source.shape[0],
                self.channels,
                self.centroids.shape[2])

        input_data = source

        # source: [N, 768, Length], reference: [N, 768, Length]
        source = source.transpose(1, 2)
        reference = reference.transpose(1, 2)
        sims = -torch.cdist(source, reference)
        best = torch.topk(sims, k, dim=2)

        result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
        result = result.transpose(1, 2)
        return result * (1-alpha) + input_data * alpha


class ContentEncoder(nn.Module):
    def __init__(self,
                 n_fft=1920,
                 hop_size=480,
                 internal_channels=256,
                 kernel_size=7,
                 num_layers=4,
                 output_channels=256,
                 hubert_channels=768,
                 num_centroids=128,
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size

        self.input_layer = nn.Sequential(
            nn.Conv1d(n_fft // 2 + 1, 80, 1),
            nn.Conv1d(80, internal_channels, 1))
        self.output_layer = nn.Conv1d(internal_channels, output_channels, 1)

        self.res_stack = nn.Sequential(
                *[ResBlock(internal_channels, kernel_size, dilation=3**i, norm=True) for i in range(num_layers)])

        self.vector_explorer = VectorExplorer(num_centroids, output_channels)
        self.to_hubert = nn.Conv1d(output_channels, hubert_channels, 1, bias=False)

    def pass_layers(self, spec):
        x = self.input_layer(spec)
        x = self.res_stack(x)
        x = self.output_layer(x)
        return x

    def forward(self, spec):
        x = self.pass_layers(spec)
        x = self.vector_explorer(x)
        return x

    def encode(self, wave):
        spec = spectrogram(wave, self.n_fft, self.hop_size)
        x = self.forward(spec)
        return x

    def predict_hubert_features(self, wave):
        spec = spectrogram(wave, self.n_fft, self.hop_size)
        z = self.pass_layers(spec)
        y1 = self.to_hubert(z)
        y2 = self.to_hubert(self.vector_explorer(z))
        return y1, y2
