#!/bin/env python
import argparse

import torch
import pytorch_lightning as pl

from opt import get_opts
from datasets import load_dataset, satellite
from metrics import load_loss, DepthLoss, SNerfLoss
from torch.utils.data import DataLoader
from collections import defaultdict

from rendering import render_rays
from models import load_model
import train_utils
import metrics
import os
import numpy as np
import datetime
from sat_utils import dsm_pointwise_abs_errors

from eval_satnerf import find_best_embbeding_for_val_image, save_nerf_output_to_images, predefined_val_ts

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class NeRF_pl(pl.LightningModule):
    """NeRF network"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.loss = load_loss(args)
        self.depth = args.ds_lambda > 0
        if self.depth:
            # depth supervision will be used
            self.depth_loss = DepthLoss(lambda_ds=args.ds_lambda)
            self.ds_drop = np.round(args.ds_drop * args.max_train_steps)
        self.define_models()
        self.val_im_dir = "{}/{}/val".format(args.logs_dir, args.exp_name)
        self.train_im_dir = "{}/{}/train".format(args.logs_dir, args.exp_name)
        self.train_steps = 0

        self.use_ts = False
        if self.args.model == "sat-nerf":
            self.loss_without_beta = SNerfLoss(lambda_sc=args.sc_lambda)
            self.use_ts = True

    def define_models(self):
        self.models = {}
        self.nerf_coarse = load_model(self.args)
        self.models['coarse'] = self.nerf_coarse
        if self.args.n_importance > 0:
            self.nerf_fine = load_model(self.args)
            self.models['fine'] = self.nerf_fine
        if self.args.model == "sat-nerf":
            self.embedding_t = torch.nn.Embedding(self.args.t_embbeding_vocab, self.args.t_embbeding_tau)
            self.models["t"] = self.embedding_t

    def forward(self, rays, ts):

        chunk_size = self.args.chunk
        batch_size = rays.shape[0]

        results = defaultdict(list)
        for i in range(0, batch_size, chunk_size):
            rendered_ray_chunks = \
                render_rays(self.models, self.args, rays[i:i + chunk_size],
                            ts[i:i + chunk_size] if ts is not None else None)
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        self.train_dataset = [] + load_dataset(self.args, split="train")
        self.val_dataset = [] + load_dataset(self.args, split="val")

    def configure_optimizers(self):

        parameters = train_utils.get_parameters(self.models)
        self.optimizer = torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=0)

        max_epochs = self.get_current_epoch(self.args.max_train_steps)
        scheduler = train_utils.get_scheduler(optimizer=self.optimizer, lr_scheduler='step', num_epochs=max_epochs)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def train_dataloader(self):
        a = DataLoader(self.train_dataset[0],
                       shuffle=True,
                       num_workers=4,
                       batch_size=self.args.batch_size,
                       pin_memory=True)
        loaders = {"color": a}
        if self.depth:
            b = DataLoader(self.train_dataset[1],
                           shuffle=True,
                           num_workers=4,
                           batch_size=self.args.batch_size,
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
        self.log("lr", train_utils.get_learning_rate(self.optimizer))
        self.train_steps += 1

        rays = batch["color"]["rays"] # (B, 11)
        rgbs = batch["color"]["rgbs"] # (B, 3)
        ts = None if not self.use_ts else batch["color"]["ts"].squeeze() # (B, 1)

        results = self(rays, ts)
        if 'beta_coarse' in results and self.get_current_epoch(self.train_steps) < 2:
            loss, loss_dict = self.loss_without_beta(results, rgbs)
        else:
            loss, loss_dict = self.loss(results, rgbs)
        self.args.noise_std *= 0.9

        if self.depth:
            tmp = self(batch["depth"]["rays"], batch["depth"]["ts"].squeeze())
            kp_depths = torch.flatten(batch["depth"]["depths"][:, 0])
            kp_weights = 1. if self.args.ds_noweights else torch.flatten(batch["depth"]["depths"][:, 1])
            loss_depth, tmp = self.depth_loss(tmp, kp_depths, kp_weights)
            if self.train_steps < self.ds_drop :
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
        if self.args.model == "sat-nerf":
            t = predefined_val_ts(batch["src_id"][0])
            ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        else:
            ts = None
        results = self(rays, ts)
        loss, loss_dict = self.loss(results, rgbs)

        self.is_validation_image = True
        if self.args.data == 'sat' and batch_nb == 0:
            self.is_validation_image = False

        typ = "fine" if "rgb_fine" in results else "coarse"
        if "h" in batch and "w" in batch:
            W, H = batch["w"], batch["h"]
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float())) # assume squared images
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        depth = train_utils.visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
        stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
        split = 'val' if self.is_validation_image else 'train'
        sample_idx = batch_nb - 1 if self.is_validation_image else batch_nb
        self.logger.experiment.add_images('{}_{}/GT_pred_depth'.format(split, sample_idx), stack, self.global_step)

        # save output for a training image (batch_nb == 0) and a validation image (batch_nb == 1)
        epoch = self.get_current_epoch(self.train_steps)
        save = not bool(epoch % self.args.save_every_n_epochs)
        if (batch_nb == 0 or batch_nb == 1) and self.args.data == 'sat' and save:
            # save some images to disk for a more detailed visualization
            out_dir = self.val_im_dir if self.is_validation_image else self.train_im_dir
            save_nerf_output_to_images(self.val_dataset[0], batch, results, out_dir, epoch)

        psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
        ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W), rgbs.view(1, 3, H, W))

        if self.args.data != 'sat':
            self.log("val/loss", loss)
            self.log("val/psnr", psnr_)
            self.log("val/ssim", ssim_)
        else:
            # 1st image is from the training set, so it must not contribute to the validation metrics
            if batch_nb != 0:
                # compute MAE
                try:
                    aoi_id = batch["src_id"][0][:7]
                    roi_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.txt")
                    gt_dsm_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.tif")
                    if os.path.exists(roi_path) and os.path.exists(gt_dsm_path):
                        depth = results[f"depth_{typ}"]
                        unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        out_path = os.path.join(self.val_im_dir, "dsm/tmp_pred_dsm_{}.tif".format(unique_identifier))
                        _ = self.val_dataset[0].get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
                        roi_metadata = np.loadtxt(roi_path)
                        if aoi_id in ["JAX_004", "JAX_260"]:
                            gt_seg_path = os.path.join(self.args.gt_dir, aoi_id + "_CLS_v2.tif")
                        else:
                            gt_seg_path = os.path.join(self.args.gt_dir, aoi_id + "_CLS.tif")
                        abs_err = dsm_pointwise_abs_errors(out_path, gt_dsm_path, roi_metadata, gt_mask_path=gt_seg_path)
                        mae_ = np.nanmean(abs_err)
                        os.remove(out_path)
                except:
                    mae_ = np.nan

                self.log("val/loss", loss)
                self.log("val/psnr", psnr_)
                self.log("val/ssim", ssim_)
                self.log("val/mae", mae_)
                for k in loss_dict.keys():
                    self.log("val/{}".format(k), loss_dict[k])

        return {"loss": loss}

    def get_current_epoch(self, tstep):
        return train_utils.get_epoch_number_from_train_step(tstep, len(self.train_dataset[0]), self.args.batch_size)

def main():

    torch.cuda.empty_cache()
    args = get_opts()
    system = NeRF_pl(args)

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir, name=args.exp_name, default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath="{}/{}".format(args.ckpts_dir, args.exp_name),
                                                 filename="{epoch:d}",
                                                 monitor="val/psnr",
                                                 mode="max",
                                                 save_top_k=-1,
                                                 every_n_val_epochs=args.save_every_n_epochs)

    trainer = pl.Trainer(max_steps=args.max_train_steps,
                         logger=logger,
                         callbacks=[ckpt_callback],
                         resume_from_checkpoint=args.ckpt_path,
                         gpus=[args.gpu_id],
                         auto_select_gpus=False,
                         deterministic=True,
                         benchmark=True,
                         weights_summary=None,
                         num_sanity_val_steps=2,
                         check_val_every_n_epoch=1,
                         profiler="simple")

    trainer.fit(system)


if __name__ == "__main__":
    main()
