# FasterSVC : FastSVC with low-latency inferencing

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

3. Train decoder and speaker encoder
```sh
python3 train_dec.py <datset.path>
```

### TIPS
- add `-fp16 True` to accelerate training with float16 if you have RTX series GPU.
- add `-b <number>` to set batch size. default is `16`.
- add `-e <number>` to set epoch. default is `1000`. 

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
python3 infer_streaming.py -i <input device id> -o <output device id> -l <loopback device id>
```
(loopback is optional)