"""
This script defines the evaluation metrics and the loss functions
"""

import torch
from kornia.losses import ssim as ssim_
import os
import shutil
from osgeo import gdal
import rasterio
import numpy as np
import datetime
from gaussian_filter import GaussianSmoothing


class MaskedMSE(torch.nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef

    def forward(self, inputs, targets, mask):
        diff = torch.flatten( ((inputs - targets) ** 2) * mask)
        return torch.sum(diff) / torch.sum(mask)


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
        self.loss_with_mask = MaskedMSE()

    def forward(self, inputs, targets):
        d = {}
        if 'mask' in inputs:
            d['c_l'] = self.loss_with_mask(inputs['rgb_coarse'], targets, inputs['mask'])
        else:
            d['c_l'] = self.loss(inputs['rgb_coarse'], targets)
        if self.lambda_s > 0:
            sun_sc = inputs['sun_sc_coarse'].squeeze()
            term2 = torch.sum(torch.square(inputs['transparency_sc_coarse'].detach() - sun_sc), -1)
            term3 = 1 - torch.sum(inputs['weights_sc_coarse'].detach() * sun_sc, -1)
            d['c_sc_term2'] = self.lambda_s * torch.mean(term2)
            d['c_sc_term3'] = self.lambda_s * torch.mean(term3)
            #d['c_sc'] = self.lambda_s * torch.mean(term2 + term3)

        if 'rgb_fine' in inputs:
            d['f_l'] = self.loss(inputs['rgb_fine'], targets)
            if self.lambda_s > 0:
                sun_sc = inputs['sun_sc_fine'].squeeze()
                term2 = torch.sum(torch.square(inputs['transparency_sc_fine'].detach() - sun_sc), -1)
                term3 = 1 - torch.sum(inputs['weights_sc_fine'].detach() * sun_sc, -1)
                d['f_sc_term2'] = self.lambda_s * torch.mean(term2)
                d['f_sc_term3'] = self.lambda_s * torch.mean(term3)
                #d['f_sc'] = self.lambda_s * torch.mean(term2 + term3)

        loss = sum(l for l in d.values())
        return loss, d

class DepthLoss(torch.nn.Module):
    def __init__(self, lambda_d=1.0):
        super().__init__()
        self.lambda_d = lambda_d
        self.loss = torch.nn.MSELoss(reduce=False)

    def forward(self, inputs, targets, weights=None):
        d = {}
        d['c_d'] = self.loss(inputs['depth_coarse'], targets)
        if 'depth_fine' in inputs:
            d['f_d'] = self.loss(inputs['depth_fine'], targets)

        if weights is None:
            for k in d.keys():
                d[k] = self.lambda_d * torch.mean(d[k])
        else:
            for k in d.keys():
                d[k] = self.lambda_d * torch.mean(weights * d[k])

        loss = sum(l for l in d.values())
        return loss, d

class SatNerfColorLoss(torch.nn.Module):
    def __init__(self, coef=1, beta_min=0.05, lambda_s=0.0):
        super().__init__()
        self.coef = coef
        self.beta_min = beta_min
        self.lambda_s = lambda_s

    def forward(self, inputs, targets):
        d = {}

        beta_coarse = torch.sum(inputs['weights_coarse'].unsqueeze(-1) * inputs['beta_coarse'], -2) + self.beta_min
        d['c_l'] = ((inputs['rgb_coarse'] - targets) ** 2 / (2 * beta_coarse ** 2)).mean()
        d['c_b'] = (3 + torch.log(beta_coarse).mean())/2  # +3 to make c_b positive since beta_min = 0.05

        if self.lambda_s > 0:
            sun_sc = inputs['sun_sc_coarse'].squeeze()
            term2 = torch.sum(torch.square(inputs['transparency_sc_coarse'].detach() - sun_sc), -1)
            term3 = 1 - torch.sum(inputs['weights_sc_coarse'].detach() * sun_sc, -1)
            d['c_sc_term2'] = self.lambda_s * torch.mean(term2)
            d['c_sc_term3'] = self.lambda_s * torch.mean(term3)

        if 'rgb_fine' in inputs:
            beta_fine = torch.sum(inputs['weights_fine'].unsqueeze(-1) * inputs['beta_fine'], -2) + self.beta_min
            d['f_l'] = ((inputs['rgb_fine'] - targets) ** 2 / (2 * beta_fine ** 2)).mean()
            d['f_b'] = (3 + torch.log(beta_fine).mean())/2  # +3 to make it positive
        loss = sum(l for l in d.values())
        return self.coef * loss, d


class CustomFilter(torch.nn.Module):

    def __init__(self, kernel_x, kernel_y):
        super(CustomFilter, self).__init__()
        self.conv_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv_x.weight = torch.nn.Parameter(torch.from_numpy(kernel_x).double().unsqueeze(0).unsqueeze(0))
        self.conv_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv_y.weight = torch.nn.Parameter(torch.from_numpy(kernel_y).double().unsqueeze(0).unsqueeze(0))

    def forward(self, tensor):
        input_g_x = self.conv_x(tensor)
        input_g_y = self.conv_y(tensor)
        return input_g_x[:,:,1:-1,:], input_g_y[:,:,:,1:-1]


class PatchesLoss(torch.nn.Module):
    def __init__(self, coef=1, beta_min=0.05):
        super().__init__()
        self.coef = coef
        self.gaussian_filter = GaussianSmoothing(channels=1, kernel_size=3, sigma=2)
        self.kernel_x = np.array([[1 / 2, 0, -1 / 2]])
        self.kernel_y = self.kernel_x.T
        self.gradient_filter = CustomFilter(self.kernel_x, self.kernel_y)

    def forward(self, inputs, patch_size, patch_real_sizes):
        d = {}

        d['c_depth_reg'] = []
        r = 0
        for tmp in patch_real_sizes:
            p_h, p_w = tmp[0], tmp[1]
            patch_depths = inputs['depth_coarse'][r:r+(p_w*p_h)].view(1, 1, p_h, p_w)
            r += patch_size**2
            input_g = self.gaussian_filter(patch_depths.double())
            dx, dy = self.gradient_filter(input_g)
            grad = torch.abs(dx) + torch.abs(dy)
            grad = torch.mean(torch.flatten(grad))
            d['c_depth_reg'].append(grad)
        d['c_depth_reg'] = 0.1 * torch.sum(torch.stack(d['c_depth_reg']))

        loss = sum(l for l in d.values())
        return self.coef * loss, d


def load_loss(args):

    loss_dict = {'nerf': ColorLoss,
                 's-nerf': SNerfLoss,
                 's-nerf-w': SNerfLoss}

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


def ssim(image_pred, image_gt):
    """
    image_pred and image_gt: (1, 3, H, W)
    important: kornia==0.5.3
    """
    return torch.mean(ssim_(image_pred, image_gt, 3))

def dsm_pointwise_abs_errors(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None, out_rdsm_path=None, out_err_path=None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_path = "tmp_crop_dsm_to_delete_{}.tif".format(unique_identifier)
    pred_rdsm_path = "tmp_crop_rdsm_to_delete_{}.tif".format(unique_identifier)

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
    if gt_mask_path is not None:
        with rasterio.open(gt_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]
        with rasterio.open(pred_dsm_path, 'w', **profile) as dst:
            pred_dsm[water_mask.astype(bool)] = np.nan
            dst.write(pred_dsm, 1)

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    with rasterio.open(pred_dsm_path, "r") as f:
        profile = f.profile
        pred_dsm = f.read()[0, :, :]

    # register and compute mae
    fix_xy = False
    try:
        import dsmr
    except:
        print("Warning: dsmr not found ! DSM registration will only use the Z dimension")
        fix_xy = True
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
    abs_err = pred_rdsm - gt_dsm

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

    return abs(abs_err)

def dsm_mae(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None):
    abs_err = dsm_pointwise_abs_errors(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=gt_mask_path)
    return np.nanmean(abs_err.ravel())
