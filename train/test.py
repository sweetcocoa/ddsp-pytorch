"""
args : 
--input : input wav
--output : output wav path
--ckpt : pretrained weight file
--config : network-corresponding yaml config file
--wave_length : wave length in format 
    (default : 0, which means all)
    WARNING : gpu memory might be not enough.
"""

import torch
import torchaudio
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from network.autoencoder.autoencoder import AutoEncoder
from omegaconf import OmegaConf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=".wav")
parser.add_argument("--output", default="output.wav")
parser.add_argument("--ckpt", default=".pth")
parser.add_argument("--config", default=".yaml")
parser.add_argument("--wave_length", default=16000)
args = parser.parse_args()

y, sr = torchaudio.load(args.input, num_frames=None if args.wave_length == 0 else args.wave_length)

config = OmegaConf.load(args.config)
if sr != config.sample_rate:
    # Resample if sampling rate is not equal to model's
    resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
    y = resampler(y)

print("File :", args.input, "Loaded")

net = AutoEncoder(config).cuda()
net.load_state_dict(torch.load(args.ckpt))
net.eval()

print("Network Loaded")

recon = net.reconstruction(y)

dereverb = recon["audio_synth"].cpu()
torchaudio.save(
    os.path.splitext(args.output)[0] + "_synth.wav", dereverb, sample_rate=config.sample_rate
)

if config.use_reverb:
    recon_add_reverb = recon["audio_reverb"].cpu()
    torchaudio.save(
        os.path.splitext(args.output)[0] + "_reverb.wav",
        recon_add_reverb,
        sample_rate=config.sample_rate,
    )
