"""
This script defines the dataloader for a dataset of multi-view satellite images
"""

import numpy as np
import os

import torch
from .satellite import SatelliteDataset
from torchvision import transforms as T

from PIL import Image
import rasterio
import rpcm
import json
import glob
import sat_utils


class SatelliteDataset_depth(SatelliteDataset):
    def __init__(self, root_dir, img_dir, split="train", img_downscale=1.0, cache_dir=None):
        """
        NeRF Satellite Dataset
        Args:
            root_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from root_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
            cache_dir: string, directory containing precomputed rays
        """
        super().__init__(root_dir, img_dir, split="train", img_downscale=1.0, cache_dir=None)


    def load_train_split(self):
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        if os.path.exists(self.json_dir + "/pts3d.npy"):
            self.tie_points = np.load(self.json_dir + "/pts3d.npy")
            self.all_rays, self.all_depths, self.all_ids = self.load_depth_data(self.json_files, self.tie_points,
                                                                                verbose=True)
        else:
            raise FileNotFoundError("Could not find {}".format(self.json_dir + "/pts3d.npy"))

    def load_depth_data(self, json_files, tie_points, verbose=False):

        all_rays, all_depths, all_sun_dirs, all_weights = [], [], [], []
        all_ids = []
        kp_weights = self.load_keypoint_weights_for_depth_supervision(json_files, tie_points)

        for t, json_p in enumerate(json_files):
            # read json
            d = sat_utils.read_dict_from_json(json_p)
            img_id = sat_utils.get_file_id(d["img"])

            if "keypoints" not in d.keys():
                raise ValueError("No 'keypoints' field was found in {}".format(json_p))

            pts2d = np.array(d["keypoints"]["2d_coordinates"])/ self.img_downscale
            pts3d = np.array(tie_points[d["keypoints"]["pts3d_indices"], :])
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)

            # build the sparse batch of rays for depth supervision
            cols, rows = pts2d.T
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            rays = sat_utils.get_rays(cols, rows, rpc, min_alt, max_alt)
            rays = self.normalize_rays(rays)
            all_rays += [rays]

            # get sun direction
            sun_dirs = self.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])
            all_sun_dirs += [sun_dirs]

            # normalize the 3d coordinates of the tie points observed in the current view
            pts3d = torch.from_numpy(pts3d).type(torch.FloatTensor)
            pts3d[:, 0] -= self.center[0]
            pts3d[:, 1] -= self.center[1]
            pts3d[:, 2] -= self.center[2]
            pts3d[:, 0] /= self.range
            pts3d[:, 1] /= self.range
            pts3d[:, 2] /= self.range

            # compute depths
            depths = torch.linalg.norm(pts3d - rays[:, :3], axis=1)
            all_depths += [depths[:, np.newaxis]]
            current_weights = torch.from_numpy(kp_weights[d["keypoints"]["pts3d_indices"]]).type(torch.FloatTensor)
            all_weights += [current_weights[:, np.newaxis]]
            if verbose:
                print("Depth {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))
            all_ids += [t * torch.ones(rays.shape[0], 1)]

        all_ids = torch.cat(all_ids, 0)
        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_depths = torch.cat(all_depths, 0)  # (len(json_files)*h*w, 1)
        all_weights = torch.cat(all_weights, 0)
        all_depths = torch.hstack([all_depths, all_weights])  # (len(json_files)*h*w, 11)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        all_depths = all_depths.type(torch.FloatTensor)
        return all_rays, all_depths, all_ids

    def load_keypoint_weights_for_depth_supervision(self, json_files, tie_points):

        n_pts = tie_points.shape[0]
        n_cams = len(json_files)
        reprojection_errors = np.zeros((n_pts, n_cams), dtype=np.float32)
        for t, json_p in enumerate(json_files):
            d = sat_utils.read_dict_from_json(json_p)

            if "keypoints" not in d.keys():
                raise ValueError("No 'keypoints' field was found in {}".format(json_p))

            pts2d = np.array(d["keypoints"]["2d_coordinates"])
            pts3d = np.array(tie_points[d["keypoints"]["pts3d_indices"], :])

            rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")

            lat, lon, alt = sat_utils.ecef_to_latlon_custom(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
            col, row = rpc.projection(lon, lat, alt)
            pts2d_reprojected = np.vstack((col, row)).T
            errs_obs_current_cam = np.linalg.norm(pts2d - pts2d_reprojected, axis=1)
            reprojection_errors[d["keypoints"]["pts3d_indices"], t] = errs_obs_current_cam

        e = np.sum(reprojection_errors, axis=1)
        e_mean = np.mean(e)
        weights = np.exp(-(e/e_mean)**2)

        return weights

    def __len__(self):
        # compute length of dataset
        if self.train:
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx):
        # take a batch from the dataset
        if self.train:
            sample = {"rays": self.all_rays[idx], "depths": self.all_depths[idx], "ts": self.all_ids[idx].long()}
        else:
            rays, depths = self.load_depth_data([self.json_files[idx]])
            ts = self.all_ids[idx] * torch.ones(rays.shape[0], 1)
            d = sat_utils.read_dict_from_json(self.json_files[idx])
            img_id = sat_utils.get_file_id(d["img"])
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            sample = {"rays": rays, "depths": depths, "ts": ts.long(), "src_id": img_id, "h": h, "w": w}
        return sample
