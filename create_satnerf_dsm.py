import glob
import numpy as np
import os
import json
import torch
import shutil
import argparse
import rasterio

from datasets import SatelliteDataset
import sat_utils
from eval_satnerf import load_nerf, batched_inference, save_nerf_output_to_images, predefined_val_ts

import warnings
warnings.filterwarnings("ignore")


def create_pretty_dsm(run_id, logs_dir, output_dir, epoch_number, checkpoints_dir=None):

    with open('{}/opts.json'.format(os.path.join(logs_dir, run_id)), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    args.root_dir = "/mnt/cdisk/roger/Datasets" + args.root_dir.split("Datasets")[-1]
    args.img_dir = "/mnt/cdisk/roger/Datasets" + args.img_dir.split("Datasets")[-1]
    args.cache_dir = "/mnt/cdisk/roger/Datasets" + args.cache_dir.split("Datasets")[-1]
    args.gt_dir = "/mnt/cdisk/roger/Datasets" + args.gt_dir.split("Datasets")[-1]

    # load pretrained nerf
    if checkpoints_dir is None:
        checkpoints_dir = args.ckpts_dir
    models = load_nerf(run_id, logs_dir, checkpoints_dir, epoch_number-1)

    # take the image closest to nadir with solar direction also closer to nadir as reference view
    reference_image = sat_utils.sort_by_increasing_view_incidence_angle(args.root_dir)[0]
    with open(sat_utils.sort_by_increasing_solar_incidence_angle(args.root_dir)[0], 'r') as f:
        d = json.load(f)
    sun_el = np.radians(float(d["sun_elevation"]))
    sun_az = np.radians(float(d["sun_azimuth"]))
    sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])

    # prepare a sat-nerf validation dataset of one single image using the reference view
    dataset = SatelliteDataset(args.root_dir, args.img_dir, split="val",
                               img_downscale=args.img_downscale, cache_dir=args.cache_dir)
    dataset.json_files = [reference_image]

    # define transient embeddings if model is sat-nerf
    if args.model == "sat-nerf":
        d_train = SatelliteDataset(args.root_dir, args.img_dir, split="train",
                                   img_downscale=args.img_downscale, cache_dir=args.cache_dir)
        if reference_image in d_train.json_files:
            t = d_train.json_files.index(reference_image)
            ts = t * torch.ones(dataset[0]["rays"].shape[0], 1).long().cuda().squeeze()
        else:
            t = predefined_val_ts(dataset[0]["src_id"][0])
            ts = t * torch.ones(dataset[0]["rays"].shape[0], 1).long().cuda().squeeze()
    else:
        ts = None

    # define nerf input and run the model
    sample = dataset[0]
    rays = sample["rays"]
    src_id = sample["src_id"]
    print(f"using image {src_id}...")
    sun_dirs = torch.from_numpy(np.tile(sun_d, (rays.shape[0], 1)))
    rays[:, 8:11] = sun_dirs.type(torch.FloatTensor)
    results = batched_inference(models, rays.cuda(), ts, args)

    # save results
    for k in sample.keys():
        if torch.is_tensor(sample[k]):
            sample[k] = sample[k].unsqueeze(0)
        else:
            sample[k] = [sample[k]]
    out_dir = os.path.join(output_dir, "pretty_dsm", run_id, "tmp")
    os.makedirs(out_dir, exist_ok=True)
    save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number)

    # clean files
    tmp_path1 = glob.glob(os.path.join(out_dir, "dsm/*.tif"))[0]
    tmp_path2 = glob.glob(os.path.join(out_dir, "gt_rgb/*.tif"))[0]
    tmp_path3 = os.path.join(output_dir, "pretty_dsm", run_id, "{}_gt_rgb.tif".format(src_id))
    pred_dsm_path = os.path.join(output_dir, "pretty_dsm", run_id, "{}_dsm_epoch{}.tif".format(src_id, epoch_number))
    shutil.copyfile(tmp_path1, pred_dsm_path)
    shutil.copyfile(tmp_path2, tmp_path3)
    shutil.rmtree(out_dir)
    for p in glob.glob(os.path.join(output_dir, "pretty_dsm", run_id, "tmp_crop*.tif")):
        os.remove(p)

    if args.gt_dir is not None:
        # evaluate NeRF generated DSM
        out_dir = os.path.join(output_dir, "pretty_dsm", run_id)
        mae = sat_utils.compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, args.gt_dir, out_dir, epoch_number)
        print("Path to output NeRF DSM: {}".format(pred_dsm_path))
        print("Altitude MAE: {}".format(np.nanmean(mae)))
        rdsm_tmp_path = os.path.join(out_dir, "{}_rdsm_epoch{}.tif".format(src_id, epoch_number))
        rdsm_path = rdsm_tmp_path.replace(".tif", "_{:.3f}.tif".format(mae))
        shutil.copyfile(rdsm_tmp_path, rdsm_path)
        os.remove(rdsm_tmp_path)

        # save tmp gt DSM
        aoi_id = src_id[:7]
        gt_dsm_path = os.path.join(args.gt_dir, "{}_DSM.tif".format(aoi_id))
        tmp_gt_path = os.path.join(output_dir, "pretty_dsm", run_id, "tmp_gt.tif")
        if aoi_id in ["JAX_004", "JAX_260"]:
            gt_seg_path = os.path.join(args.gt_dir, "{}_CLS_v2.tif".format(aoi_id))
        else:
            gt_seg_path = os.path.join(args.gt_dir, "{}_CLS.tif".format(aoi_id))
        with rasterio.open(gt_seg_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
        with rasterio.open(rdsm_path, "r") as f:
            profile = f.profile
        with rasterio.open(gt_dsm_path, "r") as f:
            gt_dsm = f.read()[0, :, :]
        with rasterio.open(tmp_gt_path, 'w', **profile) as dst:
            gt_dsm[water_mask.astype(bool)] = np.nan
            dst.write(gt_dsm, 1)

if __name__ == '__main__':
    import fire
    fire.Fire(create_pretty_dsm)