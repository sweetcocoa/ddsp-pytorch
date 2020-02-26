import torch
import torch.nn as nn

from components.harmonic_oscillator import HarmonicOscillator
from components.reverb import TrainableFIRReverb
from components.filtered_noise import FilteredNoise
from network.autoencoder.decoder import Decoder
from network.autoencoder.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, config):
        """
        encoder_config
                use_z=False, 
                sample_rate=16000,
                z_units=16,
                n_fft=2048,
                hop_length=64,
                n_mels=128,
                n_mfcc=30,
                gru_units=512
        
        decoder_config
                mlp_units=512,
                mlp_layers=3,
                use_z=False,
                z_units=16,
                n_harmonics=101,
                n_freq=65,
                gru_units=512,

        components_config
                sample_rate
                hop_length
        """
        super().__init__()

        self.decoder = Decoder(config)
        self.encoder = Encoder(config)

        hop_length = frame_length = int(config.sample_rate * config.frame_resolution)

        self.harmonic_oscillator = HarmonicOscillator(
            sr=config.sample_rate, frame_length=hop_length
        )

        self.filtered_noise = FilteredNoise(frame_length=hop_length)

        self.reverb = TrainableFIRReverb(reverb_length=config.sample_rate * 3)

        self.crepe = None
        self.config = config

    def forward(self, batch, add_reverb=True):
        """
        z

        input(dict(f0, z(optional), l)) : a dict object which contains key-values below
                f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
                z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
                loudness : torch.tensor w/ shape(B, time)
        """
        batch = self.encoder(batch)
        latent = self.decoder(batch)

        harmonic = self.harmonic_oscillator(latent)
        noise = self.filtered_noise(latent)

        audio = dict(
            harmonic=harmonic, noise=noise, audio_synth=harmonic + noise[:, : harmonic.shape[-1]]
        )

        if self.config.use_reverb and add_reverb:
            audio["audio_reverb"] = self.reverb(audio)

        audio["a"] = latent["a"]
        audio["c"] = latent["c"]

        return audio

    def get_f0(self, x, sample_rate=16000, f0_threshold=0.5):
        """
        input:
            x = torch.tensor((1), wave sample)
        
        output:
            f0 : (n_frames, ). fundamental frequencies
        """
        if self.crepe is None:
            from components.ptcrepe.ptcrepe.crepe import CREPE

            self.crepe = CREPE(self.config.crepe)
            for param in self.parameters():
                self.device = param.device
                break
            self.crepe = self.crepe.to(self.device)
        self.eval()

        with torch.no_grad():
            time, f0, confidence, activation = self.crepe.predict(
                x,
                sr=sample_rate,
                viterbi=True,
                step_size=int(self.config.frame_resolution * 1000),
                batch_size=32,
            )

            f0 = f0.float().to(self.device)
            f0[confidence < f0_threshold] = 0.0
            f0 = f0[:-1]

        return f0

    def reconstruction(self, x, sample_rate=16000, add_reverb=True, f0_threshold=0.5, f0=None):
        """
        input:
            x = torch.tensor((1), wave sample)
            f0 (if exists) = (num_frames, )

        output(dict):
            f0 : (n_frames, ). fundamental frequencies
            a : (n_frames, ). amplitudes
            c : (n_harmonics, n_frames). harmonic constants
            sig : (n_samples)
            audio_reverb : (n_samples + reverb, ). reconstructed signal
        """
        self.eval()

        with torch.no_grad():
            if f0 is None:
                f0 = self.get_f0(x, sample_rate=sample_rate, f0_threshold=f0_threshold)

            batch = dict(f0=f0.unsqueeze(0), audio=x.to(self.device),)

            recon = self.forward(batch, add_reverb=add_reverb)

            # make shape consistent(removing batch dim)
            for k, v in recon.items():
                recon[k] = v[0]

            recon["f0"] = f0

            return recon
