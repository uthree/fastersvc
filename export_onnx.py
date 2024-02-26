import argparse
import os

import torch

from module.convertor import Convertor

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outputs', default="./onnx/")
parser.add_argument('-m', '--models', default="./models/")
parser.add_argument('-opset', default=15, type=int)

args = parser.parse_args()

print("Loading models...")
convertor = Convertor()
convertor.load(args.models)

print("Exporting onnx...")
if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

opset_version = args.opset

print("Exporting pitch estimator")
fft_bin = convertor.pitch_estimator.n_fft // 2 + 1
dummy_input = torch.randn(1, fft_bin, 100)
torch.onnx.export(
        convertor.pitch_estimator,
        dummy_input,
        os.path.join(args.outputs, "pitch_estimator.onnx"),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "length"}})


print("Exporting content encoder")
fft_bin = convertor.content_encoder.n_fft // 2 + 1
dummy_input = torch.randn(1, fft_bin, 100)
torch.onnx.export(
        convertor.content_encoder,
        dummy_input,
        os.path.join(args.outputs, "content_encoder.onnx"),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "length"}})

print("Exporting decoder")
content_channels = convertor.decoder.content_channels
frames_per_second = convertor.decoder.sample_rate // convertor.decoder.frame_size
spk_dim = convertor.decoder.spk_dim
num_harmonics = convertor.decoder.num_harmonics

z = torch.randn(1, content_channels, frames_per_second) # content
e = torch.randn(1, 1, frames_per_second) # energy
spk = torch.randn(1, spk_dim, 1) # speaker embedding
src = torch.randn(1, num_harmonics+2, convertor.decoder.sample_rate) # source signal
torch.onnx.export(
        convertor.decoder,
        (z, e, spk, src),
        os.path.join(args.outputs, "decoder.onnx"),
        opset_version=opset_version,
        input_names=["content", "energy", "speaker", "source"],
        output_names=["output"],
        dynamic_axes={
            "content": {0: "batch_size", 2: "length"},
            "energy": {0: "batch_size", 2: "length"},
            "speaker": {0: "batch_size"},
            "source": {0: "batch_size", 2: "length"}})



print("Exporting speaker embedding")
idx = torch.LongTensor([0])
torch.onnx.export(
        convertor.speaker_embedding,
        idx,
        os.path.join(args.outputs, "speaker_embedding.onnx"),
        opset_version=opset_version,
        input_names=["index"],
        output_names=["output"],
        dynamic_axes={
            "index": {0: "batch_size"}})
