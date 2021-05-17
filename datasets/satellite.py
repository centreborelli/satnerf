"""
This script defines the dataloader for a dataset of multi-view satellite images
"""

import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image
import rpcm
import json


def rescale_RPC(rpc, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc model to scale
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    import copy

    rpc_scaled = copy.copy(rpc)
    rpc_scaled.row_scale *= float(alpha)
    rpc_scaled.col_scale *= float(alpha)
    rpc_scaled.row_offset *= float(alpha)
    rpc_scaled.col_offset *= float(alpha)
    return rpc_scaled


class SatelliteDataset(Dataset):

    def __init__(self, root_dir, img_dir, split='train', img_downscale=1):
        """
        root_dir: the directory where the json files with all relevant metadata per image are located
        img_dir: the directory where the satellite images are located
        split: either 'train' or 'val'
        img_downscale: downscale factor
        use_cache: during data preparation, use precomputed rays
        """
        self.json_dir = root_dir
        self.img_dir = img_dir
        self.split = split
        self.img_downscale = max(2, int(img_downscale))
        self.define_transforms()
        self.read_meta()
        self.white_back = False

    def load_data(self, json_files):
        all_rgbs, all_rays = [], []
        image_ids, image_rpcs = [], []
        for t, json_p in enumerate(json_files):
            # read json
            with open(json_p) as f:
                d = json.load(f)

            # read rgb image
            h, w = int(d["height"]), int(d["width"])
            img_p = os.path.join(self.img_dir, d["img"])
            img = Image.open(img_p)
            if self.img_downscale > 1:
                w = w//self.img_downscale
                h = h//self.img_downscale
                img = img.resize((w, h), Image.LANCZOS)
            img = self.to_tensor(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
            all_rgbs += [img]

            # get image ids and rpcs
            image_ids.append(os.path.splitext(os.path.basename(d["img"]))[0])
            current_rpc = rescale_RPC(rpcm.RPCModel(d["rpc"]), 1.0/self.img_downscale)
            image_rpcs.append(current_rpc)

            # get the unit vector of the sun light direction
            sun_el = np.radians(float(d["sun_elevation"]))
            sun_az = np.radians(float(d["sun_azimuth"]))
            sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
            # we assume all sun rays are parallel
            sun_rays_d = np.tile(sun_d, (h * w, 1))

            # we assume each ray starts at the corresponding image pixel
            # this differs from the original nerf, where each ray is built based on a origin 3d point + dir vector
            # in the satellite scenario, each ray is built based on a 2d pixel location + the rpc model
            rows, cols = np.meshgrid(np.arange(h), np.arange(w))
            rows = rows.reshape((-1, 1))
            cols = cols.reshape((-1, 1))
            cam_idx = t * np.ones(cols.shape)

            # vectors of depth limits for discretization during rendering
            nears = float(d["min_alt"]) * np.ones(cols.shape)
            fars = float(d["max_alt"]) * np.ones(cols.shape)

            rays_current_img = torch.from_numpy(np.hstack([cols, rows, cam_idx, nears, fars, sun_rays_d]))
            all_rays += [rays_current_img]  # (h*w, 8)

        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_rgbs = all_rgbs.type(all_rays.dtype)

        return all_rays, all_rgbs, image_ids, image_rpcs

    def read_meta(self):
        if self.split == "train":
            with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
                json_files = f.read().split("\n")
            json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
            self.all_rays, self.all_rgbs, self.img_ids_train, self.image_rpcs = self.load_data(json_files)
        elif self.split in ['val']: # use the first image as val image (also in train)
            with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
                json_files = f.read().split("\n")
            json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
            self.all_rays, self.all_rgbs, self.img_ids_val, self.image_rpcs = self.load_data(json_files)
        else:
            pass

    def define_transforms(self):
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return self.all_rays.shape[0]

    def __getitem__(self, idx):
        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}
        return sample
