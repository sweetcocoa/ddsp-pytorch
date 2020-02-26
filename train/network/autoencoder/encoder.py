import torch
import torchaudio
import torch.nn as nn
from components.loudness_extractor import LoudnessExtractor


class Z_Encoder(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length,
        sample_rate=16000,
        n_mels=128,
        n_mfcc=30,
        gru_units=512,
        z_units=16,
        bidirectional=False,
    ):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=20.0, f_max=8000.0,
            ),
        )

        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.permute = lambda x: x.permute(0, 2, 1)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dense = nn.Linear(gru_units * 2 if bidirectional else gru_units, z_units)

    def forward(self, batch):
        x = batch["audio"]
        x = self.mfcc(x)
        x = x[:, :, :-1]
        x = self.norm(x)
        x = self.permute(x)
        x, _ = self.gru(x)
        x = self.dense(x)
        return x


class Encoder(nn.Module):
    """
    Encoder. 

    contains: Z_encoder, loudness extractor

    Constructor arguments:
        use_z : Bool, if True, Encoder will produce z as output.
        sample_rate=16000,
        z_units=16,
        n_fft=2048,
        n_mels=128,
        n_mfcc=30,
        gru_units=512,
        bidirectional=False

    input(dict(audio, f0)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, frame)
        audio : raw audio w/ shape(B, time)

    output : a dict object which contains key-values below

        loudness : torch.tensor w/ shape(B, frame)
        f0 : same as input
        z : (optional) residual information. torch.tensor w/ shape(B, frame, z_units)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hop_length = int(config.sample_rate * config.frame_resolution)

        self.loudness_extractor = LoudnessExtractor(
            sr=config.sample_rate, frame_length=self.hop_length,
        )

        if config.use_z:
            self.z_encoder = Z_Encoder(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=self.hop_length,
                n_mels=config.n_mels,
                n_mfcc=config.n_mfcc,
                gru_units=config.gru_units,
                z_units=config.z_units,
                bidirectional=config.bidirectional,
            )

    def forward(self, batch):
        batch["loudness"] = self.loudness_extractor(batch)
        if self.config.use_z:
            batch["z"] = self.z_encoder(batch)

        if self.config.sample_rate % self.hop_length != 0:
            # if sample rate is not divided by hop_length
            # In short, this is not needed if sr == 16000
            batch["loudness"] = batch["loudness"][:, : batch["f0"].shape[-1]]
            batch["z"] = batch["z"][:, : batch["f0"].shape[-1]]

        return batch

