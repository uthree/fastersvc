import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResBlock, spectrogram


class VectorExplorer(nn.Module):
    def __init__(self, num_tokens=128, dim=32):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, dim, num_tokens))
        self.dim = dim

    def forward(self, source):
        return self.match(source)

    def match(self, source, k=4, alpha=0.0):
        reference = self.tokens.expand(
                source.shape[0],
                self.dim,
                self.tokens.shape[2])

        input_data = source

        source = source.transpose(1, 2)
        reference = reference.transpose(1, 2)
        source_norm = torch.norm(source, dim=2, keepdim=True)
        reference_norm = torch.norm(reference, dim=2, keepdim=True)
        cos_sims = torch.bmm((source / source_norm), (reference / reference_norm).transpose(1, 2))
        best = torch.topk(cos_sims, k, dim=2)

        result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
        result = result.transpose(1, 2)
        return result * (1-alpha) + input_data * alpha


class ContentEncoder(nn.Module):
    def __init__(self,
                 n_fft=1920,
                 hop_size=480,
                 internal_channels=256,
                 kernel_size=7,
                 dilations=[1, 3, 9, 1],
                 output_channels=32,
                 num_tokens=32
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size

        self.input_layer = nn.Sequential(
            nn.Conv1d(n_fft // 2 + 1, 80, 1, bias=False),
            nn.Conv1d(80, internal_channels, 1, bias=False))

        self.res_stack = nn.Sequential(
                *[ResBlock(internal_channels, kernel_size, dilation=d, norm=True) for d in dilations])

        self.output_layer = nn.Conv1d(internal_channels, output_channels, 1)
        self.vector_explorer = VectorExplorer(num_tokens, output_channels)

    def forward(self, spec):
        x = self.input_layer(spec)
        x = self.res_stack(x)
        x = self.output_layer(x)
        x = self.vector_explorer(x)
        return x

    def encode(self, wave):
        spec = spectrogram(wave, self.n_fft, self.hop_size)
        x = self.forward(spec)
        return x

    def encode_without_vector_explorer(self, wave):
        spec = spectrogram(wave, self.n_fft, self.hop_size)
        x = self.input_layer(spec)
        x = self.res_stack(x)
        x = self.output_layer(x)
        return x
