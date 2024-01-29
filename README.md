# FasterSVC: Fast voice conversion based distillated models and kNN method
(This repository is in the experimental stage. The content may change without notice.)

Other languages
 - [日本語](documents/README_ja.md)

## Model architecture
![Architecture](images/fastersvc_architecture.png)
The structure of the decoder is designed with reference to FastSVC, StreamVC, Hifi-GAN, etc.
Low latency is achieved by using a "causal" convolution layer that does not refer to future information.

## Features
- Realtime conversion
- Low latency (approximately 0.2 seconds, subject to change based on the environment and optimizations)
- Stable phase and pitch (based on the source-filter model)
- Speaker style conversion using k-nearest neighbors (kNN) method

## Requirements
- Python 3.10 or later
- PyTorch 2.0 or later with GPU environment
-  When training from scratch, prepare a large amount of human speech data (e.g., LJ Speech, JVS Corpus)

## Installation
1. clone this repository.
```sh
git clone https://github.com/uthree/fastersvc.git
```
2. install requirements
```sh
pip3 install -r requirements.txt
```

## pre-training
Train a model for basic voice conversion. At this stage, the model is not specialized for a specific speaker, but having a model that can perform basic voice synthesis allows for easy adaptation to a specific speaker with minimal adjustments.

Here are the steps:

1. Train pitch estimator.
Distill pitch estimation using a fast and parallelizable 1D CNN with the DIO algorithm from WORLD.
```sh
python3 train_pe.py <dataset path>
```

2. Train content encoder
Distill HuBERT-base. According to the WavLM paper, speaker and phoneme information are present in layers 4 and 9, respectively. Use the average of these as teacher data.
```sh
python3 train_ce.py <dataset path>
```

3. Train decoder
The goal of the decoder is to reconstruct the original waveform from pitch and content.

sh
```sh
python3 train_dec.py <datset.path>
```

### Training Options
- add `-fp16 True` to accelerate training with float16 if you have RTX series GPU.
- add `-b <number>` to set batch size. default is `16`.
- add `-e <number>` to set epoch. default is `60`.
- add `-d <device name>` to set training device, default is `cuda`.

## Inference
1. Create an directory `inputs`
2. Put audio files in `inputs`
3. Run inference script
```sh
python3 infer.py -t <target audio file>
```

## Realtime Inference with PyAudio
1. Confirm the ID of the audio device
```sh
python3 audio_device_list.py
```

2. Run inference
```sh
python3 infer_streaming.py -i <input device id> -o <output device id> -l <loopback device id> -t <target audio file>
```
(The loopback option is optional.)

## 参考文献
- [FastSVC](https://arxiv.org/abs/2011.05731)
- [kNN-VC](https://arxiv.org/abs/2305.18975)
- [WavLM](https://arxiv.org/pdf/2110.13900.pdf) (Fig. 2)
- [StreamVC](https://arxiv.org/abs/2401.03078v1)
- [Hifi-GAN](https://arxiv.org/abs/2010.05646)

This document is translated from Japanese using ChatGPT.