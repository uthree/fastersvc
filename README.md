# FasterSVC: Lightweight AI voice changer
(This repository is experimental. Contents are subject to change without notice.)

# Other languages
- [日本語](./documents/README_ja.md)

## Model structure
![Architecture](./images/fastersvc_architecture.png)
The structure of the decoder is designed with reference to FastSVC, NeuCoSVC, etc.
Low latency is achieved by using a "causal" convolution layer that does not refer to future information.

## Features
- Real-time conversion
- Low latency (approximately 0.2 seconds, may vary depending on environment and optimization.)
- Phase and pitch are stable (based on source filter model)

## Things necessary
- Python 3.10 or later
- PyTorch 2.0 or later and GPU environment
- When training with full scratch, prepare a large amount of human voice data. (LJ Speech, JVS Corpus, etc.)

## install
1. Clone this repository
```sh
git clone https://github.com/uthree/fastersvc.git
````
2. Install dependencies
```sh
pip3 install -r requirements.txt
````
## Download the pre-trained model
The model pretrained with the JVS corpus is published [here](https://huggingface.co/uthree/fastersvc-jvs-corpus-pretrained).

## Preliminary learning
Learn a model that performs basic speech conversion. At this stage, the model is not specialized for a specific speaker, but by preparing a model that can perform basic speech synthesis in advance, you can create a model that is specialized for a specific speaker with just a few adjustments. can be learned.

### Arrangement of training data
You need to arrange it as follows.
````
dataset
├────speaker0
│ ├───xxx1-xxx1.wav
│ ├───...
│ └───Lxx-0xx8.wav
└────speaker1
     ├───xx2-0xxx2.wav
     ├────...
     └────xxx7-xxx007.wav
````
Place audio from the same speaker in the same directory.

1. Pretreatment
Preprocess the dataset.
```sh
python3 preprocess.py <dataset path>
````

2. Training the pitch estimator
Pitch estimation using WORLD's harvest algorithm is distilled using a one-dimensional CNN that can be processed in high speed and in parallel.
```sh
python3 train_pe.py
````

3. Learn content encoder.
Distill HuBERT-soft.
```sh
python3 train_ce.py
````

4. Learn decoder
The decoder's goal is to reconstruct the original waveform from the pitch and content.

```sh
python3 train_dec.py
````

### Learning options
- Adding `-fp16 True` allows learning using 16-bit floating point numbers. Possible only for RTX series GPUs. However, stability may decrease.
- Change batch size with `-b <number>`. Default is `16`.
- Change epoch number with `-e <number>`. Default is `60`.
- Change the computing device with `-d <device name>`. Default is `cuda`.

## Reasoning
1. Create an `inputs` folder.
2. Put the audio file you want to convert into the `inputs` folder
3. Run the inference script
```sh
python3 infer.py -t <target audio file>
````

### Additional options
- You can set the transparency of the original audio information with `-a <number from 0.0 to 1.0>`.
- You can normalize the volume with `--normalize True`.
- You can change the calculation device with `-d <device name>`. Although it may not make much sense since it is originally high speed.
- Pitch shift can be performed with `-p <scale>`. Useful for voice conversion between men and women.

## Real-time inference with pyaudio (feature in testing stage)
1. Check the ID of your audio device
```sh
python3 audio_device_list.py
````

2. Execution
```sh
python3 infer_streaming.py -i <input device ID> -o <output device ID> -l <loopback device ID> -t <target audio file>
````
(It works even without the loopback option.)

## References
- [FastSVC](https://arxiv.org/abs/2011.05731)
- [NeuCoSVC](https://arxiv.org/abs/2312.04919)

(translated using Google Translate)