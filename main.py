#!/bin/env python
import argparse

import torch
import pytorch_lightning as pl

from opt import get_opts
from config import load_config, save_config
from datasets import load_dataset
from metrics import load_loss, DepthLoss
from torch.utils.data import DataLoader
from collections import defaultdict

from rendering import render_rays
from models import NeRF
import utils
import metrics
import os
import numpy as np


class NeRF_pl(pl.LightningModule):
    """NeRF network"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conf = load_config(args)
        self.save_hyperparameters(dict(self.conf))

        self.loss = load_loss(args)
        if self.conf.name == "s-nerf":
            self.loss.lambda_s = self.conf.lambda_s
        if self.args.depth:
            self.depth_loss = DepthLoss(coef=self.args.depthloss_lambda)
            self.depthloss_drop = np.round(self.args.depthloss_drop * self.conf.training.max_steps)
        self.define_models()
        self.val_im_dir = "{}/{}/val".format(args.logs_dir, args.exp_name)
        self.train_im_dir = "{}/{}/train".format(args.logs_dir, args.exp_name)
        self.train_steps = 0

    def define_models(self):

        self.models = {}

        self.nerf_coarse = NeRF(layers=self.conf.layers,
                                feat=self.conf.feat,
                                input_sizes=self.conf.input_sizes,
                                skips=self.conf.skips,
                                siren=self.conf.siren,
                                mapping=self.conf.mapping,
                                mapping_sizes=self.conf.mapping_sizes,
                                variant=self.conf.name)

        self.models['coarse'] = self.nerf_coarse

        if self.conf.n_importance > 0:
            self.nerf_fine = NeRF(layers=self.conf.layers,
                                  feat=self.conf.feat,
                                  input_sizes=self.conf.input_sizes,
                                  skips=self.conf.skips,
                                  siren=self.conf.siren,
                                  mapping=self.conf.mapping,
                                  mapping_sizes=self.conf.mapping_sizes,
                                  variant=self.conf.name)

            self.models['fine'] = self.nerf_fine

    def forward(self, rays):

        chunk_size = self.conf.training.chunk
        batch_size = rays.shape[0]

        results = defaultdict(list)
        for i in range(0, batch_size, chunk_size):
            rendered_ray_chunks = \
                render_rays(self.models,
                            rays[i:i + chunk_size],
                            conf=self.conf,
                            chunk=chunk_size)
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        self.train_dataset = [] + load_dataset(self.args, split="train")
        self.val_dataset = [] + load_dataset(self.args, split="val")

    def configure_optimizers(self):

        parameters = utils.get_parameters(self.models)
        self.optimizer = torch.optim.Adam(parameters,
                                          lr=self.conf.training.lr,
                                          weight_decay=self.conf.training.weight_decay)

        scheduler = utils.get_scheduler(optimizer=self.optimizer,
                                        lr_scheduler=self.conf.training.lr_scheduler,
                                        num_epochs=self.get_current_epoch(self.conf.training.max_steps))
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        a = DataLoader(self.train_dataset[0],
                       shuffle=True,
                       num_workers=4,
                       batch_size=self.conf.training.bs,
                       pin_memory=True)
        loaders = {"color": a}
        if self.args.depth:
            b = DataLoader(self.train_dataset[1],
                           shuffle=True,
                           num_workers=4,
                           batch_size=self.conf.training.bs,
                           pin_memory=True)
            loaders["depth"] = b
        return loaders

    def val_dataloader(self):
        return DataLoader(self.val_dataset[0],
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        self.log("lr", utils.get_learning_rate(self.optimizer))
        self.train_steps += 1

        rays, rgbs = batch["color"]["rays"], batch["color"]["rgbs"]
        results = self(rays)
        loss, loss_dict = self.loss(results, rgbs)
        if self.args.depth:
            tmp = self(batch["depth"]["rays"])
            kp_depths = torch.flatten(batch["depth"]["depths"][:, 0])
            kp_weights = None if self.args.depthloss_without_weights else torch.flatten(batch["depth"]["depths"][:, 1])
            loss_depth, tmp = self.depth_loss(tmp, kp_depths, kp_weights)
            if self.train_steps < self.depthloss_drop :
                loss += loss_depth
            for k in tmp.keys():
                loss_dict[k] = tmp[k]

        self.log("train/loss", loss)
        typ = "fine" if "rgb_fine" in results else "coarse"

        with torch.no_grad():
            psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
            self.log("train/psnr", psnr_)
        for k in loss_dict.keys():
            self.log("train/{}".format(k), loss_dict[k])

        self.log('train_psnr', psnr_, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):

        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        loss, loss_dict = self.loss(results, rgbs)

        typ = "fine" if "rgb_fine" in results else "coarse"
        W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        depth = utils.visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
        stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
        split = 'train' if self.args.dataset_name == 'satellite' and batch_nb == 0 else 'val'
        sample_idx = batch_nb - 1 if self.args.dataset_name == 'satellite' and batch_nb != 0 else batch_nb
        self.logger.experiment.add_images('{}_{}/GT_pred_depth'.format(split, sample_idx),
                                          stack, self.global_step)

        # save output for a training image (batch_nb == 0) and a validation image (batch_nb == 1)
        if (batch_nb == 0 or batch_nb == 1) and self.args.dataset_name == 'satellite':
            # save some images to disk for a more detailed visualization
            epoch = self.get_current_epoch(self.train_steps)
            src_path = batch["src_path"][0]
            src_id = batch["src_id"][0]
            depth = results[f"depth_{typ}"]
            out_dir = self.train_im_dir if batch_nb == 0 else self.val_im_dir
            # save depth prediction
            _, _, alts = self.val_dataset[0].get_latlonalt_from_nerf_prediction(rays.cpu(), depth.cpu())
            out_path = "{}/depth/{}_epoch{}_step{}.tif".format(out_dir, src_id, epoch, self.train_steps)
            utils.save_output_image(alts.reshape(1, H, W), out_path, src_path)
            # save dsm
            out_path = "{}/dsm/{}_epoch{}_step{}.tif".format(out_dir, src_id, epoch, self.train_steps)
            dsm = self.val_dataset[0].get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
            # save rgb image
            out_path = "{}/rgb/{}_epoch{}_step{}.tif".format(out_dir, src_id, epoch, self.train_steps)
            utils.save_output_image(img, out_path, src_path)
            # save shadow learning images
            if f"sun_{typ}" in results:
                s_v = torch.sum(results[f"weights_{typ}"] * results[f"sun_{typ}"], -1)
                out_path = "{}/sun/{}_epoch{}_step{}.tif".format(out_dir, src_id, epoch, self.train_steps)
                utils.save_output_image(s_v.cpu().reshape(1, H, W), out_path, src_path)
                rgb_albedo = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'albedo_{typ}'], -2)
                out_path = "{}/albedo/{}_epoch{}_step{}.tif".format(out_dir, src_id, epoch, self.train_steps)
                utils.save_output_image(rgb_albedo.cpu().view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)

        psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
        ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W), rgbs.view(1, 3, H, W))

        if self.args.dataset_name != 'satellite':
            self.log("val/loss", loss)
            self.log("val/psnr", psnr_)
            self.log("val/ssim", ssim_)
        else:
            # 1st image is from the training set, so it must not contribute to the validation metrics
            if batch_nb != 0:
                aoi_id = batch["src_id"][0][:7]
                roi_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.txt")
                gt_dsm_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.tif")
                if os.path.exists(roi_path) and os.path.exists(gt_dsm_path):
                    depth = results[f"depth_{typ}"]
                    out_path = os.path.join(self.val_im_dir, "dsm/tmp_pred_dsm.tif")
                    _ = self.val_dataset[0].get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
                    roi_metadata = np.loadtxt(roi_path)
                    mae_ = metrics.dsm_mae(out_path, gt_dsm_path, roi_metadata)
                    os.remove(out_path)

                self.log("val/loss", loss)
                self.log("val/psnr", psnr_)
                self.log("val/ssim", ssim_)
                self.log("val/mae", mae_)
                for k in loss_dict.keys():
                    self.log("val/{}".format(k), loss_dict[k])

        return {"loss": loss}

    def get_current_epoch(self, train_step):
        return int(train_step // (len(self.train_dataset[0]) // self.conf.training.bs))


def main():

    torch.cuda.empty_cache()
    args = get_opts()
    system = NeRF_pl(args)

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir, name=args.exp_name, default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath="{}/{}".format(args.checkpoints_dir, args.exp_name),
                                                 filename="{epoch:d}",
                                                 monitor="val/psnr",
                                                 mode="max",
                                                 save_top_k=-1)

    trainer = pl.Trainer(max_steps=system.conf.training.max_steps,
                         logger=logger,
                         callbacks=[ckpt_callback],
                         resume_from_checkpoint=args.ckpt_path,
                         gpus=[args.gpu_id],
                         auto_select_gpus=False,
                         deterministic=True,
                         benchmark=True,
                         weights_summary=None,
                         num_sanity_val_steps=1,
                         check_val_every_n_epoch=1,
                         profiler="simple")
                         #gradient_clip_val=1)

    trainer.fit(system)


if __name__ == "__main__":
    main()
