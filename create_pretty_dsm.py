import glob
import numpy as np
import os
import json
import rpcm
from eval_satnerf import load_nerf, batched_inference, save_nerf_output_to_images, find_best_embbeding_for_val_image
import torch
import shutil
from datasets import SatelliteDataset

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import rasterio
from PIL import Image
import cv2


def create_pretty_dsm(run_id, logs_dir, output_dir, epoch_number, checkpoints_dir=None):

    log_path = os.path.join(logs_dir, run_id)
    with open('{}/opts.json'.format(log_path), 'r') as f:
            args = json.load(f)

    args["root_dir"] = "/mnt/cdisk/roger/Datasets" + args["root_dir"].split("Datasets")[-1]
    args["img_dir"] = "/mnt/cdisk/roger/Datasets" + args["img_dir"].split("Datasets")[-1]
    args["cache_dir"] = "/mnt/cdisk/roger/Datasets" + args["cache_dir"].split("Datasets")[-1]
    args["gt_dir"] = "/mnt/cdisk/roger/Datasets" + args["gt_dir"].split("Datasets")[-1]
    json_paths = glob.glob(os.path.join(args["root_dir"], "*.json"))
    incidence_angles, solar_incidence_angles, solar_dir_vectors = [], [], []
    dates = []
    for json_p in json_paths:
        # read json
        with open(json_p) as f:
            d = json.load(f)

        # image incidence angle
        rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")
        c_lon, c_lat = d["geojson"]["center"][0], d["geojson"]["center"][1]
        alpha, _ = rpc.incidence_angles(c_lon, c_lat, z=0)
        incidence_angles.append(alpha)

        # solar incidence angle
        sun_el = np.radians(float(d["sun_elevation"]))
        sun_az = np.radians(float(d["sun_azimuth"]))
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        surface_normal = np.array([0., 0., 1.0])
        u1 = sun_d / np.linalg.norm(sun_d)
        u2 = surface_normal / np.linalg.norm(surface_normal)
        solar_dir_vectors.append(sun_d)
        solar_incidence_angles.append(np.arccos(np.dot(u1, u2)))

        date_str = d["acquisition_date"]
        dt_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
        date_nb = int(dt_obj.month)*10000 + int(dt_obj.day)*100 + int(dt_obj.hour)
        dates.append(date_nb)

    # take image closest to nadir with solar direction also closer to nadir as reference view
    reference_image = json_paths[np.argmin(incidence_angles)]
    # define solar direction bounds
    sun_d = solar_dir_vectors[np.argmin(solar_incidence_angles)]

    # define a sat-nerf dataset of one single image using the reference view
    dataset = SatelliteDataset(args["root_dir"], args["img_dir"], split="val",
                               img_downscale=args["img_downscale"], cache_dir=args["cache_dir"])
    dataset.json_files = [reference_image]

    # load nerf
    if checkpoints_dir is None:
        checkpoints_dir = args["checkpoints_dir"]
    models, conf = load_nerf(run_id, log_path, checkpoints_dir, epoch_number-1, args)

    # select ts if s-nerf-w
    if conf.name == "s-nerf-w" or (conf.name == "s-nerf" and args["uncertainty"]):
        d_train = SatelliteDataset(args["root_dir"], args["img_dir"], split="train",
                                   img_downscale=args["img_downscale"], cache_dir=args["cache_dir"])
        if reference_image in d_train.json_files:
            t = d_train.json_files.index(reference_image)
            rays = dataset[0]["rays"]
            ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        else:
            rays, rgbs = dataset[0]["rays"], dataset[0]["rgbs"]
            if t is None:
                train_indices = torch.unique(d_train.all_ids)
                ts = find_best_embbeding_for_val_image(models, rays.cuda(), conf, rgbs, train_indices=train_indices)
    else:
        ts = None


    # define nerf input and run the model
    sample = dataset[0]
    rays = sample["rays"]
    aoi_id = sample["src_id"][:7]
    print(f"using image {sample['src_id']}...")
    sun_dirs = torch.from_numpy(np.tile(sun_d, (rays.shape[0], 1)))
    rays[:, 8:11] = sun_dirs.type(torch.FloatTensor)
    results = batched_inference(models, rays.cuda(), ts, conf)

    # save results
    for k in sample.keys():
        if torch.is_tensor(sample[k]):
            sample[k] = sample[k].unsqueeze(0)
        else:
            sample[k] = [sample[k]]
    out_dir = os.path.join(output_dir, "pretty_dsm", run_id, "tmp")
    os.makedirs(out_dir, exist_ok=True)
    save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number)

    # evaluate NeRF generated DSM
    tmp_path = glob.glob(os.path.join(out_dir, "dsm/*.tif"))[0]
    tmp_path2 = glob.glob(os.path.join(out_dir, "gt_rgb/*.tif"))[0]
    tmp_path3 = os.path.join(output_dir, "pretty_dsm", run_id, "ref_rgb.tif")
    pred_dsm_path = os.path.join(output_dir, "pretty_dsm", run_id, "dsm_epoch{}.tif".format(epoch_number))
    pred_rdsm_path = os.path.join(output_dir, "pretty_dsm", run_id, "rdsm_epoch{}.tif".format(epoch_number))
    shutil.copyfile(tmp_path, pred_dsm_path)
    shutil.copyfile(tmp_path2, tmp_path3)
    shutil.rmtree(out_dir)

    gt_dsm_path = os.path.join(args["gt_dir"], "{}_DSM.tif".format(aoi_id))
    gt_roi_metadata = np.loadtxt(os.path.join(args["gt_dir"], "{}_DSM.txt".format(aoi_id)))
    if aoi_id in ["JAX_004", "JAX_260"]:
        gt_seg_path = os.path.join(args["gt_dir"], "{}_CLS_v2.tif".format(aoi_id))
    else:
        gt_seg_path = os.path.join(args["gt_dir"], "{}_CLS.tif".format(aoi_id))
    from eval_s2p import dsm_pointwise_abs_errors
    err_path = os.path.join(output_dir, "pretty_dsm", run_id, "rdsm_err_epoch{}.tif".format(epoch_number))
    abs_err = dsm_pointwise_abs_errors(pred_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path, out_rdsm_path=pred_rdsm_path, out_err_path=err_path)
    print("Path to output NeRF DSM: {}".format(pred_dsm_path))
    print("Altitude MAE: {}".format(np.nanmean(abs_err)))
    shutil.copyfile(pred_rdsm_path, pred_rdsm_path.replace(".tif", "_{:.3f}.tif".format(np.nanmean(abs_err))))
    os.remove(pred_rdsm_path)
    for p in glob.glob(os.path.join(output_dir, "pretty_dsm", run_id, "tmp_crop*.tif")):
        os.remove(p)

if __name__ == '__main__':
    import fire
    fire.Fire(create_pretty_dsm)