import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from datasets.satellite_depth import SatelliteDataset_depth
from datasets.satellite import get_rays
import sat_utils
import rpcm
import json
import torch
import os
import shutil
import argparse



def save_heatmap_of_reprojection_error(height, width, pts2d, track_err, smooth=20, plot=False):
    """
    Interpolate a set of tie points across height*width
    """
    from scipy.ndimage import gaussian_filter
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cols, rows = pts2d.T
    valid_pts = np.logical_and(cols < width, cols >= 0) & np.logical_and(rows < height, rows >= 0)
    pts2d, track_err = pts2d[valid_pts], track_err[valid_pts]

    # interpolate the reprojection error across the utm bbx
    all_cols, all_rows = np.meshgrid(np.arange(width), np.arange(height))
    pts2d_i = np.vstack([all_cols.ravel(), all_rows.ravel()]).T
    track_err_interp = idw_interpolation(pts2d, track_err, pts2d_i).reshape((height, width))
    track_err_interp = track_err_interp.reshape((height, width))

    # smooth the interpolation result to improve visualization
    track_err_interp = gaussian_filter(track_err_interp, sigma=smooth)

    # prepare plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")
    vmin, vmax = min(track_err), max(track_err)
    im = plt.imshow(track_err_interp, vmin=vmin, vmax=vmax)
    plt.scatter(pts2d[:, 0], pts2d[:, 1], 30, track_err, edgecolors="k", vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    n_ticks = 9
    ticks = np.linspace(vmin, vmax, n_ticks)
    cbar.set_ticks(ticks)
    tick_labels = ["{:.2f}".format(vmin + t * (vmax - vmin)) for t in np.linspace(0, 1, n_ticks)]
    tick_labels[-1] = ">=" + tick_labels[-1]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Reprojection error across AOI (pixel units)", rotation=270, labelpad=25)
    if plot:
        plt.show()
        # plt.savefig(img_path, bbox_inches="tight")
    else:
        return track_err_interp


def idw_interpolation(pts2d, z, pts2d_query, N=8):
    """
    Interpolates each query point pts2d_query from the N nearest known data points in pts2d
    each neighbor contribution follows inverse distance weighting IDW (closest points are given larger weights)
    inspired by https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python
    Example: given a query point q and N=3, finds the 3 data points nearest q at distances d1 d2 d3
             and returns the IDW average of the known values z1 z2 z3 at distances d1 d3 d3
             z(q) = (z1/d1 + z2/d2 + z3/d3) / (1/d1 + 1/d2 + 1/d3)
    Args:
        pts2d: Kx2 array, contains K 2d points whose value z is known
        z: Kx1 array, the known value of each point in pts2d
        pts2d_query: Qx2 array, contains Q 2d points that we want to interpolate
        N (optional): integer, nearest neighbours that will be employed to interpolate
    Returns:
        z_query: Qx1 array, contans the interpolated value of each input query point
    """
    from scipy.spatial import cKDTree as KDTree

    # build a KDTree using scipy, to find nearest neighbours quickly
    tree = KDTree(pts2d)

    # find the N nearest neighbours of each query point
    nn_distances, nn_indices = tree.query(pts2d_query, k=N)

    if N == 1:
        # particular case 1:
        # only one nearest neighbour to use, which is given all the weight
        z_query = z[nn_indices]
    else:
        # general case
        # interpolate by weighting the N nearest known points by 1/dist
        w = 1.0 / nn_distances
        w /= np.tile(np.sum(w, axis=1), (N, 1)).T
        z_query = np.sum(w * z[nn_indices], axis=1)

        # particular case 2:
        # the query point falls on a known point, which is given all the weight
        known_query_indices = np.where(nn_distances[:, 0] < 1e-10)[0]
        z_query[known_query_indices] = z[nn_indices[known_query_indices, 0]]
    return z_query

def check_depth_supervision_points(run_id, logs_dir, output_dir, root_dir=None, img_dir=None, gt_dir=None):

    with open('{}/opts.json'.format(os.path.join(logs_dir, run_id)), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    #args.root_dir = "/mnt/cdisk/roger/Datasets" + args.root_dir.split("Datasets")[-1]
    #args.img_dir = "/mnt/cdisk/roger/Datasets" + args.img_dir.split("Datasets")[-1]
    #args.cache_dir = "/mnt/cdisk/roger/Datasets" + args.cache_dir.split("Datasets")[-1]
    #args.gt_dir = "/mnt/cdisk/roger/Datasets" + args.gt_dir.split("Datasets")[-1]

    if gt_dir is not None:
        assert os.path.isdir(gt_dir)
        args.gt_dir = gt_dir
    if img_dir is not None:
        assert os.path.isdir(img_dir)
        args.img_dir = img_dir
    if root_dir is not None:
        assert os.path.isdir(root_dir)
        args.root_dir = root_dir
    if not os.path.isdir(args.cache_dir):
        args.cache_dir = None

    sat_dataset = SatelliteDataset_depth(root_dir=args.root_dir,
                                             img_dir=args.img_dir,
                                             split="train",
                                             cache_dir=args.cache_dir,
                                             img_downscale=args.img_downscale)

    json_files = sat_dataset.json_files
    tie_points = sat_dataset.tie_points
    kp_weights = sat_dataset.load_keypoint_weights_for_depth_supervision(json_files, tie_points)

    output_dir = os.path.join(output_dir, run_id)
    print("Output dir:", output_dir)
    for t, json_p in enumerate(json_files):
        # read json
        with open(json_p) as f:
            d = json.load(f)
        img_id = sat_utils.get_file_id(d["img"])

        if "keypoints" not in d.keys():
            raise ValueError("No 'keypoints' field was found in {}".format(json_p))

        pts2d = np.array(d["keypoints"]["2d_coordinates"]) / sat_dataset.img_downscale
        pts3d = np.array(tie_points[d["keypoints"]["pts3d_indices"], :])
        rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / sat_dataset.img_downscale)

        # build the sparse batch of rays for depth supervision
        cols, rows = pts2d.T
        min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
        rays = get_rays(cols, rows, rpc, min_alt, max_alt)
        rays = sat_dataset.normalize_rays(rays)

        # normalize the 3d coordinates of the tie points observed in the current view
        pts3d = torch.from_numpy(pts3d).type(torch.FloatTensor)
        pts3d[:, 0] -= sat_dataset.center[0]
        pts3d[:, 1] -= sat_dataset.center[1]
        pts3d[:, 2] -= sat_dataset.center[2]
        pts3d[:, 0] /= sat_dataset.range
        pts3d[:, 1] /= sat_dataset.range
        pts3d[:, 2] /= sat_dataset.range

        # compute depths
        depths = np.array(torch.linalg.norm(pts3d - rays[:, :3], axis=1))

        # retrieve weights
        current_weights = torch.from_numpy(kp_weights[d["keypoints"]["pts3d_indices"]]).type(torch.FloatTensor)

        # interpolate initial depths given by the known 3d points
        h, w = int(d["height"] // sat_dataset.img_downscale), int(d["width"] // sat_dataset.img_downscale)
        interpolated_init_depth = save_heatmap_of_reprojection_error(h, w, pts2d, depths, smooth=1, plot=False)

        # construct dsm from interpolated initial depths
        init_depth = torch.from_numpy(interpolated_init_depth).type(torch.FloatTensor)
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
        rays = sat_dataset.normalize_rays(rays)

        output_path = os.path.join(output_dir, "init_dsm_depth_supervision_{}.tif".format(img_id))
        sat_dataset.get_dsm_from_nerf_prediction(rays, init_depth.reshape((-1,1)), dsm_path=output_path)

        # crop within gt area of interest
        from osgeo import gdal
        aoi_id = img_id[:7]
        dsm_metadata = np.loadtxt(os.path.join(args.gt_dir, aoi_id + "_DSM.txt"))
        xoff, yoff = dsm_metadata[0], dsm_metadata[1]
        xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
        resolution = dsm_metadata[3]
        ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff
        ds = gdal.Open(output_path)
        crop_dsm_path = os.path.join(output_dir, "init_dsm_depth_supervision_{}_tmp.tif".format(img_id))
        ds = gdal.Translate(crop_dsm_path, ds, projWin=[ulx, uly, lrx, lry])
        ds = None
        os.remove(output_path)
        shutil.copyfile(crop_dsm_path, output_path)
        os.remove(crop_dsm_path)

        print("done {} of {}".format(t+1, len(json_files)))

if __name__ == '__main__':
    import fire
    fire.Fire(check_depth_supervision_points)
