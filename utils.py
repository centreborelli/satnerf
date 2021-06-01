"""
Additional functions
"""
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import os
import rasterio
import torch

from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, StepLR

def get_learning_rate(optimizer):
    """
    Get learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_parameters(models):
    """
    Get all model parameters recursively
    models can be a list, a dictionary or a single pytorch model
    """
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:
        # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

def get_scheduler(optimizer, lr_scheduler, num_epochs):

    eps = 1e-8
    if lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eps)
    elif lr_scheduler == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.01)
    elif lr_scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[2,4,8], gamma=0.5)
        #scheduler = MultiStepLR(optimizer, milestones=[50,100,200], gamma=0.5)
    elif lr_scheduler == 'step':
        gamma = 0.7
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    else:
        raise ValueError('lr scheduler not recognized!')

    return scheduler

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x = np.clip(x, 0, 255)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def save_output_image(input, output_path, source_path):
    """
    input: (D, H, W) where D is the number of channels (3 for rgb, 1 for grayscale)
           can be a pytorch tensor or a numpy array
    """
    # convert input to numpy array float32
    if torch.is_tensor(input):
        im_np = input.type(torch.FloatTensor).cpu().numpy()
    else:
        im_np = input.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(source_path, 'r') as src:
        profile = src.profile
        profile["dtype"] = rasterio.float32
        profile["height"] = im_np.shape[1]
        profile["width"] = im_np.shape[2]
        profile["count"] = im_np.shape[0]
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(im_np)