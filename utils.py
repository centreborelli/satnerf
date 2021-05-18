"""
Additional functions
"""
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

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
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_