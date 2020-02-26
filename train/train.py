import torch
import torch.nn as nn
import torchaudio
from omegaconf import OmegaConf
import sys, os, tqdm, glob
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from trainer.trainer import Trainer
from trainer.io import setup, set_seeds

from dataset.audiodata import SupervisedAudioData, AudioData
from network.autoencoder.autoencoder import AutoEncoder
from loss.mss_loss import MSSLoss
from optimizer.radam import RAdam

"""
"setup" allows you to OVERRIDE the config through command line interface
- for example
$ python train.py --batch_size 64 --lr 0.01 --use_reverb
"""

config = setup(default_config="../configs/violin.yaml")
# config = setup(pdb_on_error=True, trace=False, autolog=False, default_config=dict(
#     # general config
#     ckpt="../../ddsp_ckpt/violin/200131.pth",  # checkpoint
#     gpu="0",
#     num_workers=4,  # number of dataloader thread
#     seed=940513,    # random seed
#     tensorboard_dir="../tensorboard_log/",
#     experiment_name="DDSP_violin",   # experiment results are compared w/ this name.

#     # data config
#     train="../data/violin/train/",  # data directory. should contain f0, too.
#     test="../data/violin/test/",
#     waveform_sec=1.0,   # the length of training data.
#     frame_resolution=0.004,   # 1 / frame rate
#     batch_size=64,
#     f0_threshold=0.5,    # f0 with confidence below threshold will go to ZERO.
#     valid_waveform_sec=4.0,  # the length of validation data
#     n_fft=2048,    # (Z encoder)
#     n_mels=128,    # (Z encoder)
#     n_mfcc=30,     # (Z encoder)
#     sample_rate=16000,

#     # training config
#     num_step=100000,
#     validation_interval=1000,
#     lr=0.001,
#     lr_decay=0.98,
#     lr_min=1e-7,
#     lr_scheduler="multi", # 'plateau' 'no' 'cosine'
#     optimizer='radam',   # 'adam', 'radam'
#     loss="mss",
#     metric="mss",
#     resume=False,    # when training from a specific checkpoint.

#     # network config
#     mlp_units=512,
#     mlp_layers=3,
#     use_z=False,
#     use_reverb=False,
#     z_units=16,
#     n_harmonics=101,
#     n_freq=65,
#     gru_units=512,
#     crepe="full",
#     bidirectional=False,
#     ))

print(OmegaConf.create(config.__dict__).pretty())
set_seeds(config.seed)
Trainer.set_experiment_name(config.experiment_name)

net = AutoEncoder(config).cuda()

loss = MSSLoss([2048, 1024, 512, 256], use_reverb=config.use_reverb).cuda()

# Define evaluation metrics
if config.metric == "mss":

    def metric(output, gt):
        with torch.no_grad():
            return -loss(output, gt)


elif config.metric == "f0":
    # TODO Implement
    raise NotImplementedError
else:
    raise NotImplementedError
# -----------------------------/>

# Dataset & Dataloader Prepare
train_data = glob.glob(config.train + "/*.wav") * config.batch_size
train_data_csv = [
    os.path.dirname(wav)
    + f"/f0_{config.frame_resolution:.3f}/"
    + os.path.basename(os.path.splitext(wav)[0])
    + ".f0.csv"
    for wav in train_data
]

valid_data = glob.glob(config.test + "/*.wav")
valid_data_csv = [
    os.path.dirname(wav)
    + f"/f0_{config.frame_resolution:.3f}/"
    + os.path.basename(os.path.splitext(wav)[0])
    + ".f0.csv"
    for wav in valid_data
]

train_dataset = SupervisedAudioData(
    sample_rate=config.sample_rate,
    paths=train_data,
    csv_paths=train_data_csv,
    seed=config.seed,
    waveform_sec=config.waveform_sec,
    frame_resolution=config.frame_resolution,
)

valid_dataset = SupervisedAudioData(
    sample_rate=config.sample_rate,
    paths=valid_data,
    csv_paths=valid_data_csv,
    seed=config.seed,
    waveform_sec=config.valid_waveform_sec,
    frame_resolution=config.frame_resolution,
    random_sample=False,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=int(config.batch_size // (config.valid_waveform_sec / config.waveform_sec)),
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)
# -------------------------------------/>

# Setting Optimizer
if config.optimizer == "adam":
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
elif config.optimizer == "radam":
    optimizer = RAdam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
else:
    raise NotImplementedError
# -------------------------------------/>

# Setting Scheduler
if config.lr_scheduler == "cosine":
    # restart every T_0 * validation_interval steps
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=config.lr_min
    )
elif config.lr_scheduler == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=config.lr_decay
    )
elif config.lr_scheduler == "multi":
    # decay every ( 10000 // validation_interval ) steps
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [(x + 1) * 10000 // config.validation_interval for x in range(10)],
        gamma=config.lr_decay,
    )
elif config.lr_scheduler == "no":
    scheduler = None
else:
    raise ValueError(f"unknown lr_scheduler :: {config.lr_scheduler}")
# ---------------------------------------/>

trainer = Trainer(
    net,
    criterion=loss,
    metric=metric,
    train_dataloader=train_dataloader,
    val_dataloader=valid_dataloader,
    optimizer=optimizer,
    lr_scheduler=scheduler,
    ckpt=config.ckpt,
    is_data_dict=True,
    experiment_id=os.path.splitext(os.path.basename(config.ckpt))[0],
    tensorboard_dir=config.tensorboard_dir,
)

save_counter = 0
save_interval = 10


def validation_callback():
    global save_counter, save_interval
    # Save generated audio per every validation
    net.eval()

    def tensorboard_audio(data_loader, phase):

        bd = next(iter(data_loader))
        for k, v in bd.items():
            bd[k] = v.cuda()

        original_audio = bd["audio"][0]
        estimation = net(bd)

        if config.use_reverb:
            reconed_audio = estimation["audio_reverb"][0, : len(original_audio)]
            trainer.tensorboard.add_audio(
                f"{trainer.config['experiment_id']}/{phase}_recon",
                reconed_audio.cpu(),
                trainer.config["step"],
                sample_rate=config.sample_rate,
            )

        reconed_audio_dereverb = estimation["audio_synth"][0, : len(original_audio)]
        trainer.tensorboard.add_audio(
            f"{trainer.config['experiment_id']}/{phase}_recon_dereverb",
            reconed_audio_dereverb.cpu(),
            trainer.config["step"],
            sample_rate=config.sample_rate,
        )
        trainer.tensorboard.add_audio(
            f"{trainer.config['experiment_id']}/{phase}_original",
            original_audio.cpu(),
            trainer.config["step"],
            sample_rate=config.sample_rate,
        )

    tensorboard_audio(train_dataloader, phase="train")
    tensorboard_audio(valid_dataloader, phase="valid")

    save_counter += 1
    if save_counter % save_interval == 0:
        trainer.save(trainer.ckpt + f"-{trainer.config['step']}")


trainer.register_callback(validation_callback)
if config.resume:
    trainer.load(config.ckpt)

trainer.add_external_config(config)
trainer.train(step=config.num_step, validation_interval=config.validation_interval)

