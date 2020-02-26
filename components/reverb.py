"""
2020_01_17 - 2020_01_29
Simple trainable FIR reverb model for DDSP decoder.
TODO : 
    numerically stable decays
    crossfade
"""

import numpy as np
import torch
import torch.nn as nn


class TrainableFIRReverb(nn.Module):
    def __init__(self, reverb_length=48000, device="cuda"):

        super(TrainableFIRReverb, self).__init__()

        # default reverb length is set to 3sec.
        # thus this model can max out t60 to 3sec, which corresponds to rich chamber characters.
        self.reverb_length = reverb_length
        self.device = device

        # impulse response of reverb.
        self.fir = nn.Parameter(
            torch.rand(1, self.reverb_length, dtype=torch.float32).to(self.device) * 2 - 1,
            requires_grad=True,
        )

        # Initialized drywet to around 26%.
        # but equal-loudness crossfade between identity impulse and fir reverb impulse is not implemented yet.
        self.drywet = nn.Parameter(
            torch.tensor([-1.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

        # Initialized decay to 5, to make t60 = 1sec.
        self.decay = nn.Parameter(
            torch.tensor([3.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

    def forward(self, z):
        """
        Compute FIR Reverb
        Input:
            z['audio_synth'] : batch of time-domain signals
        Output:
            output_signal : batch of reverberated signals
        """

        # Send batch of input signals in time domain to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        input_signal = z["audio_synth"]
        zero_pad_input_signal = nn.functional.pad(input_signal, (0, self.fir.shape[-1] - 1))
        INPUT_SIGNAL = torch.rfft(zero_pad_input_signal, 1)

        # Build decaying impulse response and send it to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        # Dry-wet mixing is done by mixing impulse response, rather than mixing at the final stage.

        """ TODO 
        Not numerically stable decay method?
        """
        decay_envelope = torch.exp(
            -(torch.exp(self.decay) + 2)
            * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32).to(self.device)
        )
        decay_fir = self.fir * decay_envelope

        ir_identity = torch.zeros(1, decay_fir.shape[-1]).to(self.device)
        ir_identity[:, 0] = 1

        """ TODO
        Equal-loudness(intensity) crossfade between to ir.
        """
        final_fir = (
            torch.sigmoid(self.drywet) * decay_fir + (1 - torch.sigmoid(self.drywet)) * ir_identity
        )
        zero_pad_final_fir = nn.functional.pad(final_fir, (0, input_signal.shape[-1] - 1))

        FIR = torch.rfft(zero_pad_final_fir, 1)

        # Convolve and inverse FFT to get original signal.
        OUTPUT_SIGNAL = torch.zeros_like(INPUT_SIGNAL).to(self.device)
        OUTPUT_SIGNAL[:, :, 0] = (
            INPUT_SIGNAL[:, :, 0] * FIR[:, :, 0] - INPUT_SIGNAL[:, :, 1] * FIR[:, :, 1]
        )
        OUTPUT_SIGNAL[:, :, 1] = (
            INPUT_SIGNAL[:, :, 0] * FIR[:, :, 1] + INPUT_SIGNAL[:, :, 1] * FIR[:, :, 0]
        )

        output_signal = torch.irfft(OUTPUT_SIGNAL, 1)

        return output_signal
