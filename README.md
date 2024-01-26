# FasterSVC : modified FastSVC for low-latency realtime inferencing

Other languages
 - [日本語](documents/README_ja.md)

## Model architecture
![Architecture](images/fastersvc_architecture.png)
(The decoder is very similar to that of FastSVC, but with one less FiLM layer and the convolution layer changed to causal convolution.)

## Features
- streaming inference on CPU (tested on Intel Core i7-10700)
- low latency (a.c. 0.2 seconds)
- high quality (based on neural source filter model)
- kNN based style conversion
- lightweight

## Requirements
- Python
- PyTorch with GPU environment

## Installation
1. clone this repository.
```sh
git clone https://github.com/uthree/fastersvc.git
```
2. install requirements
```sh
pip3 install -r requirements.txt
```

## Training
1. Train pitch estimator.
```sh
python3 train_pe.py <dataset path>
```

2. Train content encoder.
```sh
python3 train_ce.py <dataset path>
```

3. Train decoder
```sh
python3 train_dec.py <datset.path>
```

### Training Options
- add `-fp16 True` to accelerate training with float16 if you have RTX series GPU.
- add `-b <number>` to set batch size. default is `16`.
- add `-e <number>` to set epoch. default is `1000`.
- add `-d <device name>` to set training device, default is `cuda`.

## Inference
1. create directory `inputs`
2. put audio files in `inputs`
3. run inference script
```sh
python3 infer.py -t <target audio file>
```

## Streaming inference with pyaudio
1. check your audio-device's ID
```sh
python3 audio_device_list.py
```

2. run inference
```sh
python3 infer_streaming.py -i <input device id> -o <output device id> -l <loopback device id> -t <target audio file>
```
(loopback is optional)

## References
- [FastSVC](https://arxiv.org/abs/2011.05731)
- [kNN-VC](https://arxiv.org/abs/2305.18975)
- [WavLM](https://arxiv.org/pdf/2110.13900.pdf) (Fig. 2)
