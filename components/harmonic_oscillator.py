"""
2020_01_15 - 2020_01_29
Harmonic Oscillator model for DDSP decoder.
TODO : 
    upsample + interpolation 
"""

import numpy as np
import torch
import torch.nn as nn


class HarmonicOscillator(nn.Module):
    def __init__(self, sr=16000, frame_length=64, attenuate_gain=0.02, device="cuda"):
        super(HarmonicOscillator, self).__init__()
        self.sr = sr
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain

        self.device = device

        self.framerate_to_audiorate = nn.Upsample(
            scale_factor=self.frame_length, mode="linear", align_corners=False
        )

    def forward(self, z):

        """
        Compute Addictive Synthesis
        Argument: 
            z['f0'] : fundamental frequency envelope for each sample
                - dimension (batch_num, frame_rate_time_samples)
            z['c'] : harmonic distribution of partials for each sample 
                - dimension (batch_num, partial_num, frame_rate_time_samples)
            z['a'] : loudness of entire sound for each sample
                - dimension (batch_num, frame_rate_time_samples)
        Returns:
            addictive_output : synthesized sinusoids for each sample 
                - dimension (batch_num, audio_rate_time_samples)
        """

        fundamentals = z["f0"]
        framerate_c_bank = z["c"]

        num_osc = framerate_c_bank.shape[1]

        # Build a frequency envelopes of each partials from z['f0'] data
        partial_mult = (
            torch.linspace(1, num_osc, num_osc, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        framerate_f0_bank = (
            fundamentals.unsqueeze(-1).expand(-1, -1, num_osc).transpose(1, 2) * partial_mult
        )

        # Antialias z['c']
        mask_filter = (framerate_f0_bank < self.sr / 2).float()
        antialiased_framerate_c_bank = framerate_c_bank * mask_filter

        # Upsample frequency envelopes and build phase bank
        audiorate_f0_bank = self.framerate_to_audiorate(framerate_f0_bank)
        audiorate_phase_bank = torch.cumsum(audiorate_f0_bank / self.sr, 2)

        # Upsample amplitude envelopes
        audiorate_a_bank = self.framerate_to_audiorate(antialiased_framerate_c_bank)

        # Build harmonic sinusoid bank and sum to build harmonic sound
        sinusoid_bank = (
            torch.sin(2 * np.pi * audiorate_phase_bank) * audiorate_a_bank * self.attenuate_gain
        )

        framerate_loudness = z["a"]
        audiorate_loudness = self.framerate_to_audiorate(framerate_loudness.unsqueeze(0)).squeeze(0)

        addictive_output = torch.sum(sinusoid_bank, 1) * audiorate_loudness

        return addictive_output
