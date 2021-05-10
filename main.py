#!/bin/env python
import argparse
import dataclasses
import random

import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

class Siren(torch.nn.Module):
    """Siren non-linearity"""
    def __init__():
        super().__init__()

    def forward(self, x):
        return torch.sin(30 * x)

class Mapping(torch.nn.Module):
    def __init__(mapping_size, in_size):
        super().__init__()
        B = torch.randn((mapping_size, in_size)) * 10

    def forward(self, x):
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class NeRF(pl.LightningModule):
    """Nerf network"""
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.save_hyperparameters()

        self.nl = Siren() if conf.siren else torch.nn.ReLU()

        layers = []
        in_size = sum(conf.input_sizes)
        if conf.mapping:
            self.mapping = [Mapping(map_sz, in_sz) for map_sz, in_sz in zip(conf.mapping_sizes, conf.input_sizes)]
            in_size = 2*sum(conf.mapping_sizes)

        layers.append(torch.nn.Linear(in_size, conf.feat))
        layers.append(self.nl)
        for i in range(1, conf.layers):
            layers.append(torch.nn.Linear(conf.feat, conf.feat))
            layers.append(self.nl)
        layers.append(torch.nn.Linear(conf.feat, conf.feat))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, xs):
        y = torch.cat([mapping(x) for mapping, x in zip(self.mappings, x)], dim=1)
        hid = self.net(y)
        return rgb(hid), sigma(hid)

    def configure_optimizers(self):
        beta1 = 0.5
        beta2 = 0.9

        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.conf.training.lr,
            betas=(beta1, beta2))
        
        return opt

    def training_step(self, batch, batch_idx):
        # TODO

    def validation_step(self, batch, batch_idx):
        # TODO



@dataclasses.dataclass
class TrainingConfig:
    """Sub-configuration for the training procedure."""

    lr: float = 1e-4
    bs: int = 32
    workers: int = 4

    train_batches: int = 1000
    val_batches: int = 100


@dataclasses.dataclass
class DefaultConfig:
    """Default configuration."""

    name: str = 'default'
    training: TrainingConfig = dataclasses.field(
        default_factory=TrainingConfig)

    siren: bool = True
    layers: int = 8
    feat: int = 256
    mapping: bool = True
    input_sizes: list = [3, 3]
    mapping_sizes: list = [10, 4]


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--gpu_id", type=int, nargs="*")
    parser.add_argument(
        "--checkpoint_dir",
        help="directory to save the logs and checkpoints.",
        default="checkpoints")

    args, other_args = parser.parse_known_args()

    conf = OmegaConf.structured(DefaultConfig)

    pl.seed_everything(0)

    model = NeRF(conf)
    logger = pl.loggers.TensorBoardLogger(args.checkpoint_dir, name=conf.name)

    # TODO
    cbks = [
        pl.callbacks.ModelCheckpoint(
            monitor='val_accuracy',
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{val_accuracy:.3f}", mode="max")
    ]

    # TODO
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.checkpoint_dir,
        deterministic=True,
        gpus=args.gpu_id,
        logger=logger,
        callbacks=cbks
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
