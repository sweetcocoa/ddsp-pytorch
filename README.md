# Pytorch version of DDSP

# DDSP : Differentiable Digital Signal Processing

> Original Authors : Jesse Engel, Lamtharn (Hanoi) Hantrakul, Chenjie Gu, Adam Roberts (Google)

> This Repository is NOT an official implement of authors.

## Demo Page ##

- [Link](https://sweetcocoa.github.io/ddsp-pytorch-samples/)

## How to train with your own data

1. Clone this repository

```bash
git clone https://github.com/sweetcocoa/ddsp-pytorch
```

2. Prepare your own audio data. (wav, mp3, flac.. )
3. Use ffmpeg to convert that audio's sampling rate to 16k

```bash
# example
ffmpeg -y -loglevel fatal -i $input_file -ac 1 -ar 16000 $output_file
```
4. Use [CREPE](https://github.com/marl/crepe) to precalculate the fundamental frequency of the audio.

```bash
# example
crepe directory-to-audio/ --output directory-to-audio/f0_0.004/  --viterbi --step-size 4
```

5. MAKE config file. (See configuration *config/violin.yaml* to make appropriate config file.) And edit train/train.py

```python
config = setup(default_config="../configs/your_config.yaml")
```
6. Run train/train.py

```bash
cd train
python train.py
```

## How to test your own model ##

```bash
cd train
python test.py\ 
--input input.wav\
--output output.wav\
--ckpt trained_weight.pth\
--config config/your-config.yaml\
--wave_length 16000
```

## Download pretrained weight file ###
> [download](https://github.com/sweetcocoa/ddsp-pytorch/raw/models/weight.zip)

## Contact ##

- Jongho Choi (sweetcocoa@snu.ac.kr, BS Student @ Seoul National Univ.)
- Sungho Lee (dlfqhsdugod1106@gmail.com, BS Student @ Postech.)

> Equally contributed.
