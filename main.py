#!/bin/env python
import argparse

import torch
import pytorch_lightning as pl

from opt import get_opts
from config import load_config
from datasets import load_dataset
from metrics import load_loss
from torch.utils.data import DataLoader
from collections import defaultdict

from rendering import render_rays
from models import NeRF
import utils
import metrics
import datetime

class NeRF_pl(pl.LightningModule):
    """NeRF network"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conf = load_config(args)
        self.loss = load_loss(args)
        self.define_models()
        self.save_hyperparameters('lr', 'train/loss', 'train/psnr')

    def define_models(self):

        self.nerf_coarse = NeRF(layers=self.conf.layers,
                                feat=self.conf.feat,
                                input_sizes=self.conf.input_sizes,
                                skips=self.conf.skips,
                                siren=self.conf.siren,
                                mapping=self.conf.mapping,
                                mapping_sizes=self.conf.mapping_sizes)

        self.models = [self.nerf_coarse]

        if self.conf.n_importance > 0:
            self.nerf_fine = NeRF(layers=self.conf.layers,
                                  feat=self.conf.feat,
                                  input_sizes=self.conf.input_sizes,
                                  skips=self.conf.skips,
                                  siren=self.conf.siren,
                                  mapping=self.conf.mapping,
                                  mapping_sizes=self.conf.mapping_sizes)

            self.models += [self.nerf_fine]

    def forward(self, rays, rpcs):

        chunk_size = self.conf.training.chunk
        batch_size = rays.shape[0]

        results = defaultdict(list)
        for i in range(0, batch_size, chunk_size):
            rendered_ray_chunks = \
                render_rays(self.models,
                            rays[i:i + chunk_size],
                            rpcs,
                            self.conf.n_samples,
                            self.conf.n_importance,
                            self.conf.training.use_disp,
                            self.conf.training.perturb,
                            self.conf.training.noise_std,
                            chunk_size
                            )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        self.train_dataset = load_dataset(self.args, split='train')
        #self.val_dataset = load_dataset(self.args, split='val')

    def configure_optimizers(self):

        parameters = utils.get_parameters(self.models)
        self.optimizer = torch.optim.Adam(parameters,
                                          lr=self.conf.training.lr,
                                          weight_decay=self.conf.training.weight_decay)

        return [self.optimizer]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.conf.training.bs,
                          pin_memory=True)

    def training_step(self, batch, batch_idx):
        self.log('lr', utils.get_learning_rate(self.optimizer))
        rays = batch["rays"]
        rgbs = batch["rgbs"]

        results = self(rays, self.train_dataset.image_rpcs)
        loss = self.loss(results, rgbs)
        self.log('train/loss', loss)
        self.logger.log_hyperparams(params=self.hparams, metrics={"your_metric": value})
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = metrics.psnr(results[f'rgb_{typ}'], rgbs)
            self.log('train/psnr', psnr_)

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_}}



def main():

    args = get_opts()
    system = NeRF_pl(args)

    exp_id = args.config_name if args.exp_name is None else args.exp_name
    exp_name = "{}_{}".format(exp_id, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir, name=exp_name, default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath="{}/{}".format(args.checkpoints_dir, exp_name),
                                                 filename="{epoch:d}",
                                                 monitor="val/psnr",
                                                 mode="max",
                                                 save_top_k=-1)

    trainer = pl.Trainer(max_steps=100000,
                         logger=logger,
                         callbacks=[ckpt_callback],
                         resume_from_checkpoint=args.ckpt_path,
                         gpus=1,
                         auto_select_gpus=True,
                         deterministic=True,
                         benchmark=True,
                         weights_summary=None,
                         num_sanity_val_steps=1,
                         profiler="simple")

    trainer.fit(system)


if __name__ == "__main__":
    main()
