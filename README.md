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

## Pre-training
Train a model for basic voice conversion. At this stage, the model is not specialized for a specific speaker, but having a model that can perform basic voice synthesis allows for easy adaptation to a specific speaker with minimal adjustments.

Here are the steps:

1. Train pitch estimator.
Distill pitch estimation using a fast and parallelizable 1D CNN with the harvest algorithm from WORLD.
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

## Fine-tuning
By adjusting the pre-trained model to a model specialized for conversion to a specific speaker, it is possible to create a more accurate model. This process takes much less time than pre-learning.
1. Combine only the audio files of a specific speaker into one folder.
2. Fine tune the decoder.
```sh
python3 train_dec.py <Folder containing only audio files of a specific speaker>
````
3. Create a dictionary for vector search. This eliminates the need to encode audio files each time.
```sh
python3 extract_index.py <Folder containing only audio files of a specific speaker> -o <Dictionary output destination (optional)>
```
4. When inferring, you can load arbitrary dictionary data by adding the `-idx <dictionary file>` option.

## About decoder learning
When training the decoder, you can set the weight of the log mel spectrogram loss with `--weight-mel <real number>`. (Default is `5.0`)
Adjusting this value may change the reproducibility of speaker identity.

## Training Options
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

### Additional options
- You can set the transparency of the original audio information with `-a <number from 0.0 to 1.0>`.
- You can normalize the volume with `--normalize True`.
- You can change the calculation device with `=d <device name>`. Although it may not make much sense since it is originally high speed.
- Pitch shift can be performed with `-p <scale>`. Useful for voice conversion between men and women.

### Style conversion by AdaIN (experimental feature)
By adding the `-adain True` option, you can enable the style conversion function by AdaIN. Not available for real-time inference.

## Realtime Inference with PyAudio (This is a feature in the testing stage)
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