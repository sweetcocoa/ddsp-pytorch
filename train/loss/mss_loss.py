"""
Implementation of Multi-Scale Spectral Loss as described in DDSP, 
which is originally suggested in NSF (Wang et al., 2019)
"""

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft, alpha=1.0, overlap=0.75, eps=1e-7):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)

    def forward(self, x_pred, x_true):
        S_true = self.spec(x_true)
        S_pred = self.spec(x_pred)

        linear_term = F.l1_loss(S_pred, S_true)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())

        loss = linear_term + self.alpha * log_term
        return loss


class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.

    Usage ::

    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)

    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)
    """

    def __init__(self, n_ffts: list, alpha=1.0, overlap=0.75, eps=1e-7, use_reverb=True):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts])
        if use_reverb:
            self.signal_key = "audio_reverb"
        else:
            self.signal_key = "audio_synth"

    def forward(self, x_pred, x_true):
        if isinstance(x_pred, dict):
            x_pred = x_pred[self.signal_key]

        if isinstance(x_true, dict):
            x_true = x_true["audio"]

        # cut reverbation off
        x_pred = x_pred[..., : x_true.shape[-1]]

        losses = [loss(x_pred, x_true) for loss in self.losses]
        return sum(losses).sum()

