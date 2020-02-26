import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

"""
Output : Randomly cropped wave with specific length & corresponding f0 (if necessary).
"""


class AudioData(Dataset):
    def __init__(
        self,
        paths,
        seed=940513,
        waveform_sec=4.0,
        sample_rate=16000,
        waveform_transform=None,
        label_transform=None,
    ):
        super().__init__()
        self.paths = paths
        self.random = np.random.RandomState(seed)
        self.waveform_sec = waveform_sec
        self.waveform_transform = waveform_transform
        self.label_transform = label_transform
        self.sample_rate = sample_rate

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.paths)


class SupervisedAudioData(AudioData):
    def __init__(
        self,
        paths,
        csv_paths,
        seed=940513,
        waveform_sec=1.0,
        sample_rate=16000,
        frame_resolution=0.004,
        f0_threshold=0.5,
        waveform_transform=None,
        label_transform=None,
        random_sample=True,
    ):
        super().__init__(
            paths=paths,
            seed=seed,
            waveform_sec=waveform_sec,
            sample_rate=sample_rate,
            waveform_transform=waveform_transform,
            label_transform=label_transform,
        )
        self.csv_paths = csv_paths
        self.frame_resolution = frame_resolution
        self.f0_threshold = f0_threshold
        self.num_frame = int(self.waveform_sec / self.frame_resolution)  # number of csv's row
        self.hop_length = int(self.sample_rate * frame_resolution)
        self.num_wave = int(self.sample_rate * self.waveform_sec)
        self.random_sample = random_sample

    def __getitem__(self, file_idx):
        target_f0 = pd.read_csv(self.csv_paths[file_idx])

        # sample interval
        if self.random_sample:
            idx_from = self.random.randint(
                1, len(target_f0) - self.num_frame
            )  # No samples from first frame - annoying to implement b.c it has to be padding at the first frame.
        else:
            idx_from = 1
        idx_to = idx_from + self.num_frame
        frame_from = target_f0["time"][idx_from]
        # frame_to = target_f0['time'][idx_to]
        confidence = target_f0["confidence"][idx_from:idx_to]

        f0 = target_f0["frequency"][idx_from:idx_to].values.astype(np.float32)
        f0[confidence < self.f0_threshold] = 0.0
        f0 = torch.from_numpy(f0)

        waveform_from = int(frame_from * self.sample_rate)
        # waveform_to = waveform_from + self.num_wave

        audio, sr = torchaudio.load(
            self.paths[file_idx], offset=waveform_from, num_frames=self.num_wave
        )
        audio = audio[0]
        assert sr == self.sample_rate

        return dict(audio=audio, f0=f0,)
