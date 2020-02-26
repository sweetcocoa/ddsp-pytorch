"""
2020_01_20 - 2020_01_29
Simple trainable filtered noise model for DDSP decoder.
TODO : 
    code refactoring
"""

import numpy as np
import torch
import torch.nn as nn


class FilteredNoise(nn.Module):
    def __init__(self, frame_length = 64, attenuate_gain = 1e-2, device = 'cuda'):
        super(FilteredNoise, self).__init__()
        
        self.frame_length = frame_length
        self.device = device
        self.attenuate_gain = attenuate_gain
        
    def forward(self, z):
        """
        Compute linear-phase LTI-FVR (time-varient in terms of frame by frame) filter banks in batch from network output,
        and create time-varying filtered noise by overlap-add method.
        
        Argument:
            z['H'] : filter coefficient bank for each batch, which will be used for constructing linear-phase filter.
                - dimension : (batch_num, frame_num, filter_coeff_length)
        
        """
        
        batch_num, frame_num, filter_coeff_length = z['H'].shape
        self.filter_window = nn.Parameter(torch.hann_window(filter_coeff_length * 2 - 1, dtype = torch.float32), requires_grad = False).to(self.device)
        
        INPUT_FILTER_COEFFICIENT = z['H']
        
        # Desired linear-phase filter can be obtained by time-shifting a zero-phase form (especially to a causal form to be real-time),
        # which has zero imaginery part in the frequency response. 
        # Therefore, first we create a zero-phase filter in frequency domain.
        # Then, IDFT & make it causal form. length IDFT-ed signal size can be both even or odd, 
        # but we choose odd number such that a single sample can represent the center of impulse response.
        ZERO_PHASE_FR_BANK = INPUT_FILTER_COEFFICIENT.unsqueeze(-1).expand(batch_num, frame_num, filter_coeff_length, 2).contiguous()
        ZERO_PHASE_FR_BANK[..., 1] = 0
        ZERO_PHASE_FR_BANK = ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length, 2)
        zero_phase_ir_bank = torch.irfft(ZERO_PHASE_FR_BANK, 1, signal_sizes = (filter_coeff_length * 2 - 1,))
           
        # Make linear phase causal impulse response & Hann-window it.
        # Then zero pad + DFT for linear convolution.
        linear_phase_ir_bank = zero_phase_ir_bank.roll(filter_coeff_length - 1, 1)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * self.filter_window.view(1, -1)
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(windowed_linear_phase_ir_bank, (0, self.frame_length - 1))
        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.rfft(zero_paded_windowed_linear_phase_ir_bank, 1)
        
        # Generate white noise & zero pad & DFT for linear convolution.
        noise = torch.rand(batch_num, frame_num, self.frame_length, dtype = torch.float32).view(-1, self.frame_length).to(self.device) * 2 - 1
        zero_paded_noise = nn.functional.pad(noise, (0, filter_coeff_length * 2 - 2))
        ZERO_PADED_NOISE = torch.rfft(zero_paded_noise, 1)

        # Convolve & IDFT to make filtered noise frame, for each frame, noise band, and batch.
        FILTERED_NOISE = torch.zeros_like(ZERO_PADED_NOISE).to(self.device)
        FILTERED_NOISE[:, :, 0] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0] \
            - ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1]
        FILTERED_NOISE[:, :, 1] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1] \
            + ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0]
        filtered_noise = torch.irfft(FILTERED_NOISE, 1).view(batch_num, frame_num, -1) * self.attenuate_gain         
                
        # Overlap-add to build time-varying filtered noise.
        overlap_add_filter = torch.eye(filtered_noise.shape[-1], requires_grad = False).unsqueeze(1).to(self.device)
        output_signal = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), 
                                                       overlap_add_filter, 
                                                       stride = self.frame_length, 
                                                       padding = 0).squeeze(1)
        
        return output_signal
