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

def scaling_params(v):
    vec = np.array(v)
    scale = (vec.max() - vec.min()) / 2
    offset = vec.min() + scale
    return scale, offset


def latlon_to_ecef_custom(lat, lon, alt):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z


def ecef_to_latlon_custom(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = np.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x ** 2) + (y ** 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + (ep ** 2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3)))
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt


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

    def __init__(self, root_dir, img_dir, split='train', img_downscale=1, cache_dir=None):
        """
        root_dir: the directory where the json files with all relevant metadata per image are located
        img_dir: the directory where the satellite images are located
        split: either 'train' or 'val'
        img_downscale: downscale factor
        use_cache: during data preparation, use precomputed rays
        """
        self.json_dir = root_dir
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.split = split
        self.img_downscale = max(4.0, int(img_downscale))
        self.define_transforms()
        self.read_meta()
        self.white_back = False

    def load_data(self):
        all_rgbs, all_rays = [], []
        for t, json_p in enumerate(self.json_files):

            # read json
            with open(json_p) as f:
                d = json.load(f)

            # retrieve image path and id
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = os.path.splitext(os.path.basename(d["img"]))[0]

            # get rgb colors
            rgbs = self.get_rgbs(img_p)

            # get rays
            cache_path = "{}/{}.data".format(self.cache_dir, img_id)
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = rescale_RPC(rpcm.RPCModel(d["rpc"]), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                rays = self.get_rays(h, w, rpc, min_alt, max_alt)

                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)

                # get the unit vector of the sun light direction
                sun_el = np.radians(float(d["sun_elevation"]))
                sun_az = np.radians(float(d["sun_azimuth"]))
                sun_d = np.array([np.sin(sun_az) * np.cos(sun_el),
                                  np.cos(sun_az) * np.cos(sun_el),
                                  np.sin(sun_el)])
                sun_rays_d = np.tile(sun_d, (rays.shape[0], 1))

            all_rgbs += [rgbs]
            all_rays += [rays]  # (h*w, 11)
            print(t)

        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 11)
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_rays = all_rays.type(torch.FloatTensor)
        all_rgbs = all_rgbs.type(torch.FloatTensor)
        return all_rays, all_rgbs

    def get_rgbs(self, img_path):

        # read rgb colors
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if self.img_downscale > 1:
            w = int(w // self.img_downscale)
            h = int(h // self.img_downscale)
            img = img.resize((w, h), Image.LANCZOS)
        img = self.to_tensor(img)  # (3, h, w)
        rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
        rgbs = rgbs.type(torch.FloatTensor)
        return rgbs

    def get_rays(self, h, w, rpc, min_alt, max_alt):

        # each ray is built based on a origin 3d point + dir vector
        # in the satellite scenario we find the lower bound of the ray by localizing the image pixel at min alt
        # the upper bound of the ray is found by localizing the image pixel at max alt
        # the director vector goes results from lower_bound - upper bound
        rows, cols = np.meshgrid(np.arange(h), np.arange(w))
        rows = rows.reshape((-1,))
        cols = cols.reshape((-1,))
        min_alts = float(min_alt) * np.ones(cols.shape)
        max_alts = float(max_alt) * np.ones(cols.shape)
        lons, lats = rpc.localization(cols, rows, min_alts)
        x_near, y_near, z_near = latlon_to_ecef_custom(lats, lons, min_alts)
        xyz_near = np.vstack([x_near, y_near, z_near]).T
        lons, lats = rpc.localization(cols, rows, max_alts)
        x_far, y_far, z_far = latlon_to_ecef_custom(lats, lons, max_alts)
        xyz_far = np.vstack([x_far, y_far, z_far]).T
        d = xyz_far - xyz_near

        fars = np.linalg.norm(d, axis=1)
        nears = float(0) * np.ones(fars.shape)

        rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
        rays_o = xyz_near

        nears, fars = nears[:, np.newaxis], fars[:, np.newaxis]
        rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears, fars]))
        rays = rays.type(torch.FloatTensor)
        return rays

    def read_meta(self):
        if self.split == "train":
            with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
                json_files = f.read().split("\n")
            self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
            self.all_rays, self.all_rgbs = self.load_data()

            self.all_rays[:, 0] -= torch.mean(self.all_rays[:, 0])
            self.all_rays[:, 1] -= torch.mean(self.all_rays[:, 1])
            self.all_rays[:, 2] -= torch.mean(self.all_rays[:, 2])

        elif self.split == 'val':
            with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
                json_files = f.read().split("\n")
            self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        else:
            pass

    def define_transforms(self):
        self.to_tensor = T.ToTensor()

    def __len__(self):
        if self.split == "train":
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx):

        if self.split == "train":
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        else:
            json_p = self.json_files[idx]

            with open(json_p) as f:
                d = json.load(f)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = os.path.splitext(os.path.basename(d["img"]))[0]

            # get colors
            rgbs = self.get_rgbs(img_p)
            cache_path = "{}/{}.data".format(self.cache_dir, img_id)
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                # get rays
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = rescale_RPC(rpcm.RPCModel(d["rpc"]), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                rays = self.get_rays(h, w, rpc, min_alt, max_alt)

                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)

            sample = {'rays': rays,
                      'rgbs': rgbs}
        return sample
