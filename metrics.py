"""
This script defines the evaluation metrics and the loss functions
"""

import torch
from kornia.losses import ssim as dssim

class ColorLoss(torch.nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss

class SNerfLoss(torch.nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.lambda_s = 0.05

    def forward(self, inputs, targets):
        term1 = self.loss(inputs['rgb_coarse'], targets)
        term2 = torch.square(inputs['transparency_coarse'].detach() - inputs['sun_visibility_coarse']).sum(1)
        term3 = 1 - (inputs['weights_coarse'].detach() * inputs['sun_visibility_coarse']).sum(1)
        loss = term1 + self.lambda_s * torch.mean(term2 + term3)

        if 'rgb_fine' in inputs:
            term1 = self.loss(inputs['rgb_fine'], targets)
            term2 = torch.square(inputs['transparency_fine'].detach() - inputs['sun_visibility_fine']).sum(1)
            term3 = 1 - (inputs['weights_fine'].detach() * inputs['sun_visibility_fine']).sum(1)
            loss += term1 + self.lambda_s * torch.mean(term2 + term3)

        return self.coef * loss

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
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]