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
import glob


def get_file_id(filename):
    """
    return what is left after removing directory and extension from a path
    """
    return os.path.splitext(os.path.basename(filename))[0]


def scaling_params(v):
    """
    find the scale and offset of a vector
    """
    vec = np.array(v)
    scale = (vec.max() - vec.min()) / 2
    offset = vec.min() + scale
    return scale, offset


def utm_from_latlon(lats, lons):
    """
    convert lat-lon to utm
    """
    import pyproj
    import utm
    from pyproj import Transformer

    n = utm.latlon_to_zone_number(lats[0], lons[0])
    l = utm.latitude_to_zone_letter(lats[0])
    proj_src = pyproj.Proj("+proj=latlong")
    proj_dst = pyproj.Proj("+proj=utm +zone={}{}".format(n, l))
    transformer = Transformer.from_proj(proj_src, proj_dst)
    easts, norths = transformer.transform(lons, lats)
    #easts, norths = pyproj.transform(proj_src, proj_dst, lons, lats)
    return easts, norths


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
    def __init__(self, root_dir, img_dir, split="train", img_downscale=1.0, cache_dir=None, depth=False):
        """
        NeRF Satellite Dataset
        Args:
            root_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from root_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
            use_cache: string, directory containing precomputed rays
        """
        self.json_dir = root_dir
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.split = split
        self.img_downscale = float(img_downscale)
        self.img_to_tensor = T.ToTensor()
        self.white_back = False
        self.depth = depth

        assert os.path.exists(root_dir), "root_dir does not exist"
        assert os.path.exists(img_dir), "img_dir does not exist"

        # load scene center and range
        if not os.path.exists("{}/scene.loc".format(self.json_dir)):
            print("Could not find a scene.loc file in the root directory, creating one...")
            print("Warning: this can take some minutes")
            all_json = glob.glob("{}/*.json".format(self.json_dir))
            for json_p in all_json:
                all_rays = []
                with open(json_p) as f:
                    d = json.load(f)
                    h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                    rpc = rescale_RPC(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                    min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                    cols, rows = np.meshgrid(np.arange(h), np.arange(w))
                    rays = self.get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                    all_rays += [rays]
                all_rays = torch.cat(all_rays, 0)
            d = {}

            near_points = all_rays[:,:3]
            far_points = all_rays[:,:3] + all_rays[:,7:8] * all_rays[:,3:6]
            all_points = torch.cat([near_points, far_points], 0)
            d["X_scale"], d["X_offset"] = scaling_params(all_points[:, 0])
            d["Y_scale"], d["Y_offset"] = scaling_params(all_points[:, 1])
            d["Z_scale"], d["Z_offset"] = scaling_params(all_points[:, 2])

            with open("{}/scene.loc".format(self.json_dir), "w") as f:
                json.dump(d, f, indent=2)
            print("... done !")
        with open(os.path.join(self.json_dir, "scene.loc")) as f:
            d = json.load(f)
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))

        # load dataset split
        if self.split == "train":
            with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
                json_files = f.read().split("\n")
            self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
            if self.depth:
                if os.path.exists(self.json_dir + "/pts3d.npy"):
                    self.tie_points = np.load(self.json_dir + "/pts3d.npy")
                    self.all_rays, self.all_depths = self.load_depth_data(self.json_files, self.tie_points, verbose=True)
                else:
                    raise FileNotFoundError("Could not find {}".format(self.json_dir + "/pts3d.npy"))
            else:
                self.all_rays, self.all_rgbs = self.load_data(self.json_files, verbose=True)
        elif self.split == "val":
            with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
                json_files = f.read().split("\n")
            self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
            # add an extra image from the training set to the validation set (for debugging purposes)
            with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
                json_files = f.read().split("\n")
            self.json_files = [os.path.join(self.json_dir, json_files[0])] + self.json_files
        else:
            pass

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
        all_rgbs, all_rays, all_sun_dirs = [], [], []
        for t, json_p in enumerate(json_files):

            # read json
            with open(json_p) as f:
                d = json.load(f)

            # retrieve image path and id
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = get_file_id(d["img"])

            # get rgb colors
            rgbs = self.get_rgbs(img_p)

            # get rays
            cache_path = "{}/{}.data".format(self.cache_dir, img_id)
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = rescale_RPC(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])

                # create grid with all pixel coordinates and compute rays
                cols, rows = np.meshgrid(np.arange(h), np.arange(w))
                rays = self.get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)

            # normalize rays
            rays = self.normalize_rays(rays)

            # get sun direction
            sun_dirs = self.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]
            if verbose:
                print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))

        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        all_rgbs = all_rgbs.type(torch.FloatTensor)
        return all_rays, all_rgbs

    def load_depth_data(self, json_files, tie_points, verbose=False):

        all_rays, all_depths, all_sun_dirs, all_weights = [], [], [], []
        kp_weights = self.load_keypoint_weights_for_depth_supervision(json_files, tie_points)

        for t, json_p in enumerate(json_files):
            # read json
            with open(json_p) as f:
                d = json.load(f)
            img_id = get_file_id(d["img"])

            if "keypoints" not in d.keys():
                raise ValueError("No 'keypoints' field was found in {}".format(json_p))

            pts2d = np.array(d["keypoints"]["2d_coordinates"])/ self.img_downscale
            pts3d = np.array(tie_points[d["keypoints"]["pts3d_indices"], :])
            rpc = rescale_RPC(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)

            # build the sparse batch of rays for depth supervision
            cols, rows = pts2d.T
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            rays = self.get_rays(cols, rows, rpc, min_alt, max_alt)
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

        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_depths = torch.cat(all_depths, 0)  # (len(json_files)*h*w, 1)
        all_weights = torch.cat(all_weights, 0)
        all_depths = torch.hstack([all_depths, all_weights])  # (len(json_files)*h*w, 11)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        all_depths = all_depths.type(torch.FloatTensor)

        return all_rays, all_depths

    def load_keypoint_weights_for_depth_supervision(self, json_files, tie_points):

        n_pts = tie_points.shape[0]
        n_cams = len(json_files)
        reprojection_errors = np.zeros((n_pts, n_cams), dtype=np.float32)
        for t, json_p in enumerate(json_files):
            with open(json_p) as f:
                d = json.load(f)

            if "keypoints" not in d.keys():
                raise ValueError("No 'keypoints' field was found in {}".format(json_p))

            pts2d = np.array(d["keypoints"]["2d_coordinates"])
            pts3d = np.array(tie_points[d["keypoints"]["pts3d_indices"], :])

            rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")

            lat, lon, alt = ecef_to_latlon_custom(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
            col, row = rpc.projection(lon, lat, alt)
            pts2d_reprojected = np.vstack((col, row)).T
            errs_obs_current_cam = np.linalg.norm(pts2d - pts2d_reprojected, axis=1)
            reprojection_errors[d["keypoints"]["pts3d_indices"], t] = errs_obs_current_cam

        e = np.sum(reprojection_errors, axis=1)
        e_mean = np.mean(e)
        weights = np.exp(-(e/e_mean)**2)

        return weights

    def get_rgbs(self, img_path):
        """
        Read rgb values from an image
        Args:
            img_path: string, path to the input image
        Returns:
            rgb: (h*w, 3) tensor of floats encoding h*w rgb colors
        """
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if self.img_downscale > 1:
            w = int(w // self.img_downscale)
            h = int(h // self.img_downscale)
            img = img.resize((w, h), Image.LANCZOS)
        img = self.img_to_tensor(img)  # (3, h, w)
        rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
        rgbs = rgbs.type(torch.FloatTensor)
        return rgbs

    def get_rays(self, cols, rows, rpc, min_alt, max_alt):
        """
        Draw a set of rays from a satellite image
        Each ray is defined by an origin 3d point + a direction vector
        First the bounds of each ray are found by localizing each pixel at min and max altitude
        Then the corresponding direction vector is found by the difference between such bounds
        Args:
            cols: 1d array with image column coordinates
            rows: 1d array with image row coordinates
            rpc: RPC model with the localization function associated to the satellite image
            min_alt: float, the minimum altitude observed in the image
            max_alt: float, the maximum altitude observed in the image
        Returns:
            rays: (h*w, 8) tensor of floats encoding h*w rays
                  columns 0,1,2 correspond to the rays origin
                  columns 3,4,5 correspond to the direction vector
                  columns 6,7 correspond to the distance of the ray bounds with respect to the camera
        """

        min_alts = float(min_alt) * np.ones(cols.shape)
        max_alts = float(max_alt) * np.ones(cols.shape)

        # assume the points of maximum altitude are those closest to the camera
        lons, lats = rpc.localization(cols, rows, max_alts)
        x_near, y_near, z_near = latlon_to_ecef_custom(lats, lons, max_alts)
        xyz_near = np.vstack([x_near, y_near, z_near]).T

        # similarly, the points of minimum altitude are the furthest away from the camera
        lons, lats = rpc.localization(cols, rows, min_alts)
        x_far, y_far, z_far = latlon_to_ecef_custom(lats, lons, min_alts)
        xyz_far = np.vstack([x_far, y_far, z_far]).T

        # define the rays origin as the nearest point coordinates
        rays_o = xyz_near

        # define the unit direction vector
        d = xyz_far - xyz_near
        rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

        # assume the nearest points are at distance 0 from the camera
        # the furthest points are at distance Euclidean distance(far - near)
        fars = np.linalg.norm(d, axis=1)
        nears = float(0) * np.ones(fars.shape)

        # create a stack with the rays origin, direction vector and near-far bounds
        rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
        rays = rays.type(torch.FloatTensor)
        return rays

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
        lats, lons, alts = ecef_to_latlon_custom(xyz[:, 0], xyz[:, 1], xyz[:, 2])
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
        easts, norths = utm_from_latlon(lats, lons)
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
        dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=3, sigma=float("inf"))

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
        if self.split == "train":
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx):
        # take a batch from the dataset
        if self.split == "train":
            if self.depth:
                sample = {"rays": self.all_rays[idx], "depths": self.all_depths[idx]}
            else:
                sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}
        else:
            json_p = self.json_files[idx]
            with open(json_p) as f:
                d = json.load(f)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = get_file_id(d["img"])

            if self.depth:
                rays, depths = self.load_depth_data([json_p])
                sample = {"rays": rays, "depths": depths, "src_path": img_p, "src_id": img_id}
            else:
                rays, rgbs = self.load_data([json_p])
                sample = {"rays": rays, "rgbs": rgbs, "src_path": img_p, "src_id": img_id}
        return sample
