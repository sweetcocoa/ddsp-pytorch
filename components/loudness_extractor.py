"""
2020_01_29 - 2020_02_03
Loudness Extractor / Envelope Follower
TODO :
    check appropriate gain structure
    GPU test
"""

import numpy as np
import torch
import torch.nn as nn


class LoudnessExtractor(nn.Module):
    def __init__(self,
                 sr = 16000,
                 frame_length = 64,
                 attenuate_gain = 2.,
                 device = 'cuda'):
        
        super(LoudnessExtractor, self).__init__()

        self.sr = sr
        self.frame_length = frame_length
        self.n_fft = self.frame_length * 5
        
        self.device = device
        
        self.attenuate_gain = attenuate_gain
        self.smoothing_window = nn.Parameter(torch.hann_window(self.n_fft, dtype = torch.float32), requires_grad = False).to(self.device)

    

    def torch_A_weighting(self, FREQUENCIES, min_db = -45.0):
        """
        Compute A-weighting weights in Decibel scale (codes from librosa) and 
        transform into amplitude domain (with DB-SPL equation).
        
        Argument: 
            FREQUENCIES : tensor of frequencies to return amplitude weight
            min_db : mininum decibel weight. appropriate min_db value is important, as 
                exp/log calculation might raise numeric error with float32 type. 
        
        Returns:
            weights : tensor of amplitude attenuation weights corresponding to the FREQUENCIES tensor.
        """
        
        # Calculate A-weighting in Decibel scale.
        FREQUENCY_SQUARED = FREQUENCIES ** 2 
        const = torch.tensor([12200, 20.6, 107.7, 737.9]) ** 2.0
        WEIGHTS_IN_DB = 2.0 + 20.0 * (torch.log10(const[0]) + 4 * torch.log10(FREQUENCIES)
                               - torch.log10(FREQUENCY_SQUARED + const[0])
                               - torch.log10(FREQUENCY_SQUARED + const[1])
                               - 0.5 * torch.log10(FREQUENCY_SQUARED + const[2])
                               - 0.5 * torch.log10(FREQUENCY_SQUARED + const[3]))
        
        # Set minimum Decibel weight.
        if min_db is not None:
            WEIGHTS_IN_DB = torch.max(WEIGHTS_IN_DB, torch.tensor([min_db], dtype = torch.float32).to(self.device))
        
        # Transform Decibel scale weight to amplitude scale weight.
        weights = torch.exp(torch.log(torch.tensor([10.], dtype = torch.float32).to(self.device)) * WEIGHTS_IN_DB / 10) 
        
        return weights

        
    def forward(self, z):
        """
        Compute A-weighted Loudness Extraction
        Input:
            z['audio'] : batch of time-domain signals
        Output:
            output_signal : batch of reverberated signals
        """
        
        input_signal = z['audio']
        paded_input_signal = nn.functional.pad(input_signal, (self.frame_length * 2, self.frame_length * 2))
        sliced_signal = paded_input_signal.unfold(1, self.n_fft, self.frame_length)
        sliced_windowed_signal = sliced_signal * self.smoothing_window
        
        SLICED_SIGNAL = torch.rfft(sliced_windowed_signal, 1, onesided = False)
        
        SLICED_SIGNAL_LOUDNESS_SPECTRUM = torch.zeros(SLICED_SIGNAL.shape[:-1])
        SLICED_SIGNAL_LOUDNESS_SPECTRUM = SLICED_SIGNAL[:, :, :, 0] ** 2 + SLICED_SIGNAL[:, :, :, 1] ** 2
                
        freq_bin_size = self.sr / self.n_fft
        FREQUENCIES = torch.tensor([(freq_bin_size * i) % (0.5 * self.sr) for i in range(self.n_fft)]).to(self.device)
        A_WEIGHTS = self.torch_A_weighting(FREQUENCIES)
        
        A_WEIGHTED_SLICED_SIGNAL_LOUDNESS_SPECTRUM = SLICED_SIGNAL_LOUDNESS_SPECTRUM * A_WEIGHTS
        A_WEIGHTED_SLICED_SIGNAL_LOUDNESS = torch.sqrt(torch.sum(A_WEIGHTED_SLICED_SIGNAL_LOUDNESS_SPECTRUM, 2)) / self.n_fft * self.attenuate_gain
        
        return A_WEIGHTED_SLICED_SIGNAL_LOUDNESS
