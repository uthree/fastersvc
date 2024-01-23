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
dummy_input = torch.randn(1, 961, 100)
torch.onnx.export(
        convertor.pitch_estimator,
        dummy_input,
        os.path.join(args.outputs, "pitch_estimator.onnx"),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "length"}})


print("Exporting speaker encoder")
dummy_input = torch.randn(1, 961, 100)
torch.onnx.export(
        convertor.speaker_encoder,
        dummy_input,
        os.path.join(args.outputs, "speaker_encoder.onnx"),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "length"}})


print("Exporting content encoder")
dummy_input = torch.randn(1, 961, 100)
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
z = torch.randn(1, 32, 100) # content
p = torch.randn(1, 1, 100) # pitch
e = torch.randn(1, 1, 100) # energy
spk = torch.randn(1, 256, 1) # speaker
src = torch.randn(1, 16, 48000) # source signal
torch.onnx.export(
        convertor.decoder,
        (z, p, e, spk, src),
        os.path.join(args.outputs, "decoder.onnx"),
        opset_version=opset_version,
        input_names=["content", "pitch", "energy", "speaker", "source"],
        output_names=["output"],
        dynamic_axes={
            "content": {0: "batch_size", 2: "length"},
            "pitch": {0: "batch_size", 2: "length"},
            "energy": {0: "batch_size", 2: "length"},
            "source": {0: "batch_size", 2: "length"}})
