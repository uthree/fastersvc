import argparse
import os

import torch

from module.convertor import Convertor
from module.index import IndexForOnnx

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outputs', default="./onnx/")
parser.add_argument('-m', '--models', default="./models/")
parser.add_argument('-idx', '--index', default='NONE')
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

print("Exporting Decoder")
content_channels = convertor.decoder.content_channels
frames_per_second = convertor.decoder.sample_rate // convertor.decoder.frame_size
z = torch.randn(1, content_channels, frames_per_second) # content
p = torch.randn(1, 1, frames_per_second) # pitch
e = torch.randn(1, 1, frames_per_second) # energy
src = torch.randn(1, 1, convertor.decoder.sample_rate) # source signal
torch.onnx.export(
        convertor.decoder,
        (z, p, e, src),
        os.path.join(args.outputs, "decoder.onnx"),
        opset_version=opset_version,
        input_names=["content", "pitch", "energy",  "source"],
        output_names=["output"],
        dynamic_axes={
            "content": {0: "batch_size", 2: "length"},
            "pitch": {0: "batch_size", 2: "length"},
            "energy": {0: "batch_size", 2: "length"},
            "source": {0: "batch_size", 2: "length"}})


if args.index != 'NONE':
    print("Exporting Index")
    vectors = torch.load(args.index)
    index_matcher = IndexForOnnx(vectors)
    z = torch.randn(1, content_channels, frames_per_second)
    torch.onnx.export(
            index_matcher,
            (z,),
            os.path.join(args.outputs, "index.onnx"),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "length"}})
