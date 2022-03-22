"""
This script defines the dataloader for a dataset of multi-view satellite images
"""

import numpy as np
import os

import torch
from torch.utils.data import Dataset


import rasterio
import rpcm
import glob
import sat_utils

class SatelliteDataset(Dataset):
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
        self.json_dir = root_dir
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.train = split == "train"
        self.img_downscale = float(img_downscale)
        self.white_back = False

        assert os.path.exists(root_dir), f"root_dir {root_dir} does not exist"
        assert os.path.exists(img_dir), f"img_dir {img_dir} does not exist"

        # load scaling params
        if not os.path.exists(f"{self.json_dir}/scene.loc"):
            self.init_scaling_params()
        d = sat_utils.read_dict_from_json(os.path.join(self.json_dir, "scene.loc"))
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))

        # load dataset split
        if self.train:
            self.load_train_split()
        else:
            self.load_val_split()

    def load_train_split(self):
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        self.all_rays, self.all_rgbs, self.all_ids = self.load_data(self.json_files, verbose=True)

    def load_val_split(self):
        with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        # add an extra image from the training set to the validation set (for debugging purposes)
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        n_train_ims = len(json_files)
        self.all_ids = [i + n_train_ims for i, j in enumerate(self.json_files)]
        self.json_files = [os.path.join(self.json_dir, json_files[0])] + self.json_files
        self.all_ids = [0] + self.all_ids

    def init_scaling_params(self):
        print("Could not find a scene.loc file in the root directory, creating one...")
        print("Warning: this can take some minutes")
        all_json = glob.glob("{}/*.json".format(self.json_dir))
        all_rays = []
        for json_p in all_json:
            d = sat_utils.read_dict_from_json(json_p)
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays = sat_utils.get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
            all_rays += [rays]
        all_rays = torch.cat(all_rays, 0)
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_params(all_points[:, 2])
        sat_utils.write_dict_to_json(d, f"{self.json_dir}/scene.loc")
        print("... done !")

    def load_data(self, json_files, verbose=False):
        """
        Load all relevant information from a set of json files
        Args:
            json_files: list containing the path to the input json files
        Returns:
            all_rays: (N, 11) tensor of floats encoding all ray-related parameters corresponding to N rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
                      columns 8,9,10 correspond to the sun direction vectors
            all_rgbs: (N, 3) tensor of floats encoding all the rgb colors corresponding to N rays
        """
        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []
        for t, json_p in enumerate(json_files):

            # read json, image path and id
            d = sat_utils.read_dict_from_json(json_p)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = sat_utils.get_file_id(d["img"])

            # get rgb colors
            rgbs = sat_utils.load_tensor_from_rgb_geotiff(img_p, self.img_downscale)

            # get rays
            cache_path = "{}/{}.data".format(self.cache_dir, img_id)
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                rays = sat_utils.get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)
            rays = self.normalize_rays(rays)

            # get sun direction
            sun_dirs = self.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]
            if verbose:
                print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))

        all_ids = torch.cat(all_ids, 0)
        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        all_rgbs = all_rgbs.type(torch.FloatTensor)
        return all_rays, all_rgbs, all_ids

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        rays[:, 6] /= self.range
        rays[:, 7] /= self.range
        return rays

    def get_sun_dirs(self, sun_elevation_deg, sun_azimuth_deg, n_rays):
        """
        Get sun direction vectors
        Args:
            sun_elevation_deg: float, sun elevation in  degrees
            sun_azimuth_deg: float, sun azimuth in degrees
            n_rays: number of rays affected by the same sun direction
        Returns:
            sun_d: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
        """
        sun_el = np.radians(sun_elevation_deg)
        sun_az = np.radians(sun_azimuth_deg)
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
        sun_dirs = sun_dirs.type(torch.FloatTensor)
        return sun_dirs

    def get_latlonalt_from_nerf_prediction(self, rays, depth):
        """
        Compute an image of altitudes from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
        Returns:
            lats: numpy vector of length h*w with the latitudes of the predicted points
            lons: numpy vector of length h*w with the longitude of the predicted points
            alts: numpy vector of length h*w with the altitudes of the predicted points
        """

        # convert inputs to double (avoids loss of resolution later when the tensors are converted to numpy)
        rays = rays.double()
        depth = depth.double()

        # use input rays + predicted sigma to construct a point cloud
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        xyz_n = rays_o + rays_d * depth.view(-1, 1)

        # denormalize prediction to obtain ECEF coordinates
        xyz = xyz_n * self.range
        xyz[:, 0] += self.center[0]
        xyz[:, 1] += self.center[1]
        xyz[:, 2] += self.center[2]

        # convert to lat-lon-alt
        xyz = xyz.data.numpy()
        lats, lons, alts = sat_utils.ecef_to_latlon_custom(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        return lats, lons, alts

    def get_dsm_from_nerf_prediction(self, rays, depth, dsm_path=None, roi_txt=None):
        """
        Compute a DSM from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
            dsm_path (optional): string, path to output DSM, in case you want to write it to disk
            roi_txt (optional): compute the DSM only within the bounds of the region of interest of the txt
        Returns:
            dsm: (h, w) numpy array with the output dsm
        """

        # get point cloud from nerf depth prediction
        lats, lons, alts = self.get_latlonalt_from_nerf_prediction(rays, depth)
        easts, norths = sat_utils.utm_from_latlon(lats, lons)
        cloud = np.vstack([easts, norths, alts]).T

        # (optional) read region of interest, where lidar GT is available
        if roi_txt is not None:
            gt_roi_metadata = np.loadtxt(roi_txt)
            xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
            xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
            resolution = gt_roi_metadata[3]
            yoff += ysize * resolution  # weird but seems necessary ?
        else:
            resolution = 0.5
            xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
            ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
            xoff = np.floor(xmin / resolution) * resolution
            xsize = int(1 + np.floor((xmax - xoff) / resolution))
            yoff = np.ceil(ymax / resolution) * resolution
            ysize = int(1 - np.floor((ymin - yoff) / resolution))

        from plyflatten import plyflatten
        from plyflatten.utils import rasterio_crs, crs_proj
        import utm
        import affine
        import rasterio

        # run plyflatten
        dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf"))

        n = utm.latlon_to_zone_number(lats[0], lons[0])
        l = utm.latitude_to_zone_letter(lats[0])
        crs_proj = rasterio_crs(crs_proj("{}{}".format(n, l), crs_type="UTM"))

        # (optional) write dsm to disk
        if dsm_path is not None:
            os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
            profile = {}
            profile["dtype"] = dsm.dtype
            profile["height"] = dsm.shape[0]
            profile["width"] = dsm.shape[1]
            profile["count"] = 1
            profile["driver"] = "GTiff"
            profile["nodata"] = float("nan")
            profile["crs"] = crs_proj
            profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
            with rasterio.open(dsm_path, "w", **profile) as f:
                f.write(dsm[:, :, 0], 1)

        return dsm

    def __len__(self):
        # compute length of dataset
        if self.train:
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx):
        # take a batch from the dataset
        if self.train:
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx], "ts": self.all_ids[idx].long()}
        else:
            rays, rgbs, _ = self.load_data([self.json_files[idx]])
            ts = self.all_ids[idx] * torch.ones(rays.shape[0], 1)
            d = sat_utils.read_dict_from_json(self.json_files[idx])
            img_id = sat_utils.get_file_id(d["img"])
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            sample = {"rays": rays, "rgbs": rgbs, "ts": ts.long(), "src_id": img_id, "h": h, "w": w}
        return sample
