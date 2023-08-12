"""
This script defines the evaluation metrics and the loss functions
"""

import torch
from kornia.losses import ssim as ssim_

class NerfLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_dict = {}
        loss_dict['coarse_color'] = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss_dict['fine_color'] = self.loss(inputs['rgb_fine'], targets)

        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

def uncertainty_aware_loss(loss_dict, inputs, gt_rgb, typ, beta_min=0.05):
    beta = torch.sum(inputs[f'weights_{typ}'].unsqueeze(-1) * inputs['beta_coarse'], -2) + beta_min
    loss_dict[f'{typ}_color'] = ((inputs[f'rgb_{typ}'] - gt_rgb) ** 2 / (2 * beta ** 2)).mean()
    loss_dict[f'{typ}_logbeta'] = (3 + torch.log(beta).mean()) / 2  # +3 to make c_b positive since beta_min = 0.05
    return loss_dict

def solar_correction(loss_dict, inputs, typ, lambda_sc=0.05):
    # computes the solar correction terms defined in Shadow NeRF and adds them to the dictionary of losses
    sun_sc = inputs[f'sun_sc_{typ}'].squeeze()
    term2 = torch.sum(torch.square(inputs[f'transparency_sc_{typ}'].detach() - sun_sc), -1)
    term3 = 1 - torch.sum(inputs[f'weights_sc_{typ}'].detach() * sun_sc, -1)
    loss_dict[f'{typ}_sc_term2'] = lambda_sc/3. * torch.mean(term2)
    loss_dict[f'{typ}_sc_term3'] = lambda_sc/3. * torch.mean(term3)
    return loss_dict

class SNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.05):
        super().__init__()
        self.lambda_sc = lambda_sc
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_dict = {}
        typ = 'coarse'
        loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
        if self.lambda_sc > 0:
            loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
            if self.lambda_sc > 0:
                loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

class SatNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.0):
        super().__init__()
        self.lambda_sc = lambda_sc

    def forward(self, inputs, targets):
        loss_dict = {}
        typ = 'coarse'
        loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)
        if self.lambda_sc > 0:
            loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)
            if self.lambda_sc > 0:
                loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

class DepthLoss(torch.nn.Module):
    def __init__(self, lambda_ds=1.0):
        super().__init__()
        self.lambda_ds = lambda_ds/3.
        self.loss = torch.nn.MSELoss(reduce=False)

    def forward(self, inputs, targets, weights=1.):
        loss_dict = {}
        typ = 'coarse'
        loss_dict[f'{typ}_ds'] = self.loss(inputs['depth_coarse'], targets)
        if 'depth_fine' in inputs:
            typ = 'fine'
            loss_dict[f'{typ}_ds'] = self.loss(inputs['depth_fine'], targets)
        # apply weights
        for k in loss_dict.keys():
            loss_dict[k] = self.lambda_ds * torch.mean(weights * loss_dict[k])
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

def load_loss(args):
    if args.model == "nerf":
        loss_function = NerfLoss()
    elif args.model == "s-nerf":
        loss_function = SNerfLoss(lambda_sc=args.sc_lambda)
    elif args.model == "sat-nerf":
        loss_function = SatNerfLoss(lambda_sc=args.sc_lambda)
    else:
        raise ValueError(f'model {args.model} is not valid')
    return loss_function

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt):
    """
    image_pred and image_gt: (1, 3, H, W)
    important: kornia==0.5.3
    """
    return torch.mean(ssim_(image_pred, image_gt, 3))
