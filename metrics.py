"""
This script defines the evaluation metrics and the loss functions
"""

import torch
from kornia.losses import ssim as ssim_
import os
import shutil
import gdal
import rasterio
import numpy as np

class ColorLoss(torch.nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        d = {}
        d['c_l'] = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            d['f_l'] = self.loss(inputs['rgb_fine'], targets)

        loss = sum(l for l in d.values())
        return self.coef * loss, d

class SNerfLoss(torch.nn.Module):
    def __init__(self, lambda_s=0.05):
        super().__init__()
        self.lambda_s = lambda_s
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        d = {}
        d['c_l'] = self.loss(inputs['rgb_coarse'], targets)
        if self.lambda_s > 0:
            term2 = torch.square(inputs['trans_sc_coarse'].detach() - inputs['sun_sc_coarse']).sum(1)
            term3 = 1 - (inputs['weights_sc_coarse'].detach() * inputs['sun_sc_coarse']).sum(1)
            d['c_sc'] = self.lambda_s * torch.mean(term2 + term3)

        if 'rgb_fine' in inputs:
            d['f_l'] = self.loss(inputs['rgb_fine'], targets)
            if self.lambda_s > 0:
                term2 = torch.square(inputs['trans_sc_fine'].detach() - inputs['sun_sc_fine']).sum(1)
                term3 = 1 - (inputs['weights_sc_fine'].detach() * inputs['sun_sc_fine']).sum(1)
                d['f_sc'] = self.lambda_s * torch.mean(term2 + term3)

        loss = sum(l for l in d.values())
        return loss, d

class DepthLoss(torch.nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = torch.nn.MSELoss(reduce=False)

    def forward(self, inputs, targets, weights=None):
        d = {}
        d['c_d'] = self.loss(inputs['depth_coarse'], targets)
        if 'depth_fine' in inputs:
            d['f_d'] = self.loss(inputs['depth_fine'], targets)

        if weights is None:
            for k in d.keys():
                d[k] = torch.mean(d[k])
        else:
            for k in d.keys():
                d[k] = torch.mean(weights * d[k])

        loss = sum(l for l in d.values())
        return self.coef * loss, d

def load_loss(args):

    loss_dict = {'nerf': ColorLoss,
                 's-nerf': SNerfLoss}

    loss_function = loss_dict[args.config_name]()
    return loss_function


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    important: kornia==0.5.3
    """
    return torch.mean(ssim_(image_pred, image_gt, 3))

def dsm_pointwise_abs_errors(in_dsm_path, gt_dsm_path, dsm_metadata, out_rdsm_path=None, out_err_path=None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    pred_dsm_path = "tmp_crop_dsm_to_delete.tif"
    pred_rdsm_path = "tmp_crop_rdsm_to_delete.tif"

    # read dsm metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution = dsm_metadata[3]

    # define projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # crop predicted dsm using gdal translate
    ds = gdal.Open(in_dsm_path)
    ds = gdal.Translate(pred_dsm_path, ds, projWin=[ulx, uly, lrx, lry])
    ds = None
    # os.system("gdal_translate -projwin {} {} {} {} {} {}".format(ulx, uly, lrx, lry, source_path, crop_path))

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    with rasterio.open(pred_dsm_path, "r") as f:
        profile = f.profile
        pred_dsm = f.read()[0, :, :]

    # register and compute mae
    fix_xy = False
    if fix_xy:
        pred_rdsm = pred_dsm + np.nanmean((gt_dsm - pred_dsm).ravel())
        with rasterio.open(pred_rdsm_path, 'w', **profile) as dst:
            dst.write(pred_rdsm, 1)
    else:
        import dsmr
        transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)
        dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)
        with rasterio.open(pred_rdsm_path, "r") as f:
            pred_rdsm = f.read()[0, :, :]
    abs_err = abs(pred_rdsm - gt_dsm)

    # remove tmp files and write output tifs if desired
    os.remove(pred_dsm_path)
    if out_rdsm_path is not None:
        if os.path.exists(out_rdsm_path):
            os.remove(out_rdsm_path)
        os.makedirs(os.path.dirname(out_rdsm_path), exist_ok=True)
        shutil.copyfile(pred_rdsm_path, out_rdsm_path)
    os.remove(pred_rdsm_path)
    if out_err_path is not None:
        if os.path.exists(out_err_path):
            os.remove(out_err_path)
        os.makedirs(os.path.dirname(out_err_path), exist_ok=True)
        with rasterio.open(out_err_path, 'w', **profile) as dst:
            dst.write(abs_err, 1)

    return abs_err

def dsm_mae(in_dsm_path, gt_dsm_path, dsm_metadata):
    abs_err = dsm_pointwise_abs_errors(in_dsm_path, gt_dsm_path, dsm_metadata)
    return np.nanmean(abs_err.ravel())
