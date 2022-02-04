#!/bin/env python
import argparse

import torch
import pytorch_lightning as pl

from opt import get_opts
from config import load_config, save_config
from datasets import load_dataset
from metrics import load_loss, DepthLoss, SatNerfColorLoss, PatchesLoss
from torch.utils.data import DataLoader
from collections import defaultdict

from rendering import render_rays
from models import NeRF
import utils
import metrics
import os
import numpy as np

from eval_aoi import find_best_embbeding_for_val_image, save_nerf_output_to_images, predefined_val_ts

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class NeRF_pl(pl.LightningModule):
    """NeRF network"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conf = load_config(args)
        self.save_hyperparameters(dict(self.conf))

        self.loss = load_loss(args)
        if "s-nerf" in self.conf.name:
            self.loss.lambda_s = self.conf.lambda_s
        if self.args.patches:
            self.patches_loss = PatchesLoss()
        elif self.args.depth:
            self.depth_loss = DepthLoss(lambda_d=self.args.depthloss_lambda)
            self.depthloss_drop = np.round(self.args.depthloss_drop * self.conf.training.max_steps)
        self.t_embbeding_size = self.conf.N_tau if "N_tau" in dict(self.conf).keys() else 0
        self.define_models()
        if self.conf.name == "s-nerf-w" and self.models["coarse"].predict_uncertainty:
            self.loss = SatNerfColorLoss(lambda_s=self.conf.lambda_s)
        self.val_im_dir = "{}/{}/val".format(args.logs_dir, args.exp_name)
        self.train_im_dir = "{}/{}/train".format(args.logs_dir, args.exp_name)
        self.train_steps = 0

        if self.conf.name == "s-nerf-w":
            self.embedding_t = torch.nn.Embedding(self.conf.N_vocab, self.conf.N_tau)
            self.models["t"] = self.embedding_t

    def define_models(self):

        self.models = {}

        self.nerf_coarse = NeRF(layers=self.conf.layers,
                                feat=self.conf.feat,
                                input_sizes=self.conf.input_sizes,
                                skips=self.conf.skips,
                                siren=self.conf.siren,
                                mapping=self.conf.mapping,
                                mapping_sizes=self.conf.mapping_sizes,
                                variant=self.conf.name,
                                t_embedding_dims=self.t_embbeding_size,
                                predict_uncertainty=self.args.uncertainty)

        self.models['coarse'] = self.nerf_coarse

        if self.conf.n_importance > 0:
            self.nerf_fine = NeRF(layers=self.conf.layers,
                                  feat=self.conf.feat,
                                  input_sizes=self.conf.input_sizes,
                                  skips=self.conf.skips,
                                  siren=self.conf.siren,
                                  mapping=self.conf.mapping,
                                  mapping_sizes=self.conf.mapping_sizes,
                                  variant=self.conf.name,
                                  t_embedding_dims=self.t_embbeding_size,
                                  predict_uncertainty=self.args.uncertainty)

            self.models['fine'] = self.nerf_fine

    def forward(self, rays, ts):

        chunk_size = self.conf.training.chunk
        batch_size = rays.shape[0]

        results = defaultdict(list)
        for i in range(0, batch_size, chunk_size):
            rendered_ray_chunks = \
                render_rays(self.models, self.conf, rays[i:i + chunk_size],
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

        parameters = utils.get_parameters(self.models)
        self.optimizer = torch.optim.Adam(parameters,
                                          lr=self.conf.training.lr,
                                          weight_decay=self.conf.training.weight_decay)

        scheduler = utils.get_scheduler(optimizer=self.optimizer,
                                        lr_scheduler=self.conf.training.lr_scheduler,
                                        num_epochs=self.get_current_epoch(self.conf.training.max_steps))
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
                       batch_size=self.conf.training.bs,
                       pin_memory=True)
        loaders = {"color": a}
        if self.args.patches:
            b = DataLoader(self.train_dataset[1],
                           shuffle=True,
                           num_workers=4,
                           batch_size=int(self.conf.training.bs//(self.args.patch_size**2)),
                           pin_memory=True)
            loaders["patches"] = b
        elif self.args.depth:
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

        rays = batch["color"]["rays"] # (B, 11)
        rgbs = batch["color"]["rgbs"] # (B, 3)

        if self.conf.name == "s-nerf-w":
            ts = batch["color"]["ts"].squeeze() # (B, 1)
        else:
            ts = None

        results = self(rays, ts)
        if "mask" in batch["color"]:
            results["mask"] = batch["color"]["mask"]
        loss, loss_dict = self.loss(results, rgbs)

        if self.args.patches:
            # remove the batch dimension
            rays_p, rgbs_p = batch["patches"]["rays"], batch["patches"]["rgbs"]
            ts_p = batch["patches"]["ts"]
            rays_p = rays_p.reshape((-1, rays_p.shape[-1]))  # (B * patch_size**2, 11)
            ts_p = ts_p.reshape((-1, ts_p.shape[-1]))  # (B * patch_size**2, 1)
            tmp_ = self(rays_p, ts_p.squeeze())
            patch_lengths = batch["patches"]["patch_size"]
            loss_patches, tmp = self.patches_loss(tmp_, self.args.patch_size, patch_lengths)
            loss += loss_patches
            for k in tmp.keys():
                loss_dict[k] = tmp[k]
        elif self.args.depth:
            tmp = self(batch["depth"]["rays"], batch["depth"]["ts"].squeeze())
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
        if self.conf.name == "s-nerf-w":
            t = predefined_val_ts(batch["src_id"][0])
            if t is None:
                ts = find_best_embbeding_for_val_image(self.models, rays, self.conf, rgbs)
            else:
                ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        else:
            ts = None
        results = self(rays, ts)
        if "mask" in batch:
            mask = batch["mask"].view(-1, 1)
            results["mask"] = torch.cat([mask, mask, mask], 1)
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
        epoch = self.get_current_epoch(self.train_steps)
        save = not bool(epoch % self.args.save_every_n_epochs)
        if (batch_nb == 0 or batch_nb == 1) and self.args.dataset_name == 'satellite' and save:
            # save some images to disk for a more detailed visualization
            out_dir = self.train_im_dir if batch_nb == 0 else self.val_im_dir
            save_nerf_output_to_images(self.val_dataset[0], batch, results, out_dir, epoch)

        psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
        ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W), rgbs.view(1, 3, H, W))

        if self.args.dataset_name != 'satellite':
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
                        out_path = os.path.join(self.val_im_dir, "dsm/tmp_pred_dsm.tif")
                        _ = self.val_dataset[0].get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
                        roi_metadata = np.loadtxt(roi_path)
                        if aoi_id in ["JAX_004", "JAX_260"]:
                            gt_seg_path = os.path.join(self.args.gt_dir, aoi_id + "_CLS_v2.tif")
                        else:
                            gt_seg_path = os.path.join(self.args.gt_dir, aoi_id + "_CLS.tif")
                        mae_ = metrics.dsm_mae(out_path, gt_dsm_path, roi_metadata, gt_mask_path=gt_seg_path)
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
                                                 save_top_k=-1,
                                                 every_n_val_epochs=args.save_every_n_epochs)

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
