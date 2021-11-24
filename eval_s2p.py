import os
import subprocess
import json
import numpy as np
import random
import glob
import rasterio
import shutil
import datetime
from osgeo import gdal

def dsm_pointwise_abs_errors(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None, out_rdsm_path=None, out_err_path=None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_path = "tmp_crop_dsm_to_delete_{}.tif".format(unique_identifier)
    pred_rdsm_path = "tmp_crop_rdsm_to_delete_{}.tif".format(unique_identifier)

    # read dsm metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution = dsm_metadata[3]

    # define projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # crop predicted dsm using gdal translate
    ds = gdal.Open(in_dsm_path)
    ds = gdal.Translate(pred_dsm_path, ds, projWin=[ulx, uly, lrx, lry])
    ds = None
    # os.system("gdal_translate -projwin {} {} {} {} {} {}".format(ulx, uly, lrx, lry, source_path, crop_path))
    if gt_mask_path is not None:
        with rasterio.open(gt_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]
        with rasterio.open(pred_dsm_path, 'w', **profile) as dst:
            pred_dsm[water_mask.astype(bool)] = np.nan
            dst.write(pred_dsm, 1)

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    with rasterio.open(pred_dsm_path, "r") as f:
        profile = f.profile
        pred_dsm = f.read()[0, :, :]

    # register and compute mae
    fix_xy = False
    try:
        import dsmr
    except:
        print("Warning: dsmr not found ! DSM registration will only use the Z dimension")
        fix_xy = True
    if fix_xy:
        pred_rdsm = pred_dsm + np.nanmean((gt_dsm - pred_dsm).ravel())
        with rasterio.open(pred_rdsm_path, 'w', **profile) as dst:
            dst.write(pred_rdsm, 1)
    else:
        import dsmr
        transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)
        dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)
        with rasterio.open(pred_rdsm_path, "r") as f:
            pred_rdsm = f.read()[0, :, :]
    abs_err = abs(pred_rdsm - gt_dsm)

    # remove tmp files and write output tifs if desired
    os.remove(pred_dsm_path)
    if out_rdsm_path is not None:
        if os.path.exists(out_rdsm_path):
            os.remove(out_rdsm_path)
        os.makedirs(os.path.dirname(out_rdsm_path), exist_ok=True)
        shutil.copyfile(pred_rdsm_path, out_rdsm_path)
    os.remove(pred_rdsm_path)
    if out_err_path is not None:
        if os.path.exists(out_err_path):
            os.remove(out_err_path)
        os.makedirs(os.path.dirname(out_err_path), exist_ok=True)
        with rasterio.open(out_err_path, 'w', **profile) as dst:
            dst.write(abs_err, 1)

    return abs_err

#######################################################################################
#######################################################################################
# script starts here

def select_pairs(root_dir, n_pairs=1):

    # load paths of the json files in the training split
    #with open(os.path.join(root_dir, "val.txt"), "r") as f:
    #    json_files = f.read().split("\n")
    #json_paths = [os.path.join(root_dir, bn) for bn in json_files]
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    n_train = len(json_paths)

    # list all possible pairs of training samples
    remaining_pairs = []
    n_possible_pairs = 0
    for i in np.arange(n_train):
        for j in np.arange(i + 1, n_train):
            remaining_pairs.append((i, j))
            n_possible_pairs += 1

    # select a random pairs
    selected_pairs_idx, selected_pairs_json_paths = [], []
    for idx in range(n_pairs):
        selected_pairs_idx.append(random.choice(remaining_pairs))
        i, j = selected_pairs_idx[-1][0], selected_pairs_idx[-1][1]
        selected_pairs_json_paths.append((json_paths[i], json_paths[j]))
        remaining_pairs = list(set(remaining_pairs) - set(selected_pairs_idx))

    return selected_pairs_json_paths, n_possible_pairs


def run_s2p(json_path_l, json_path_r, img_dir, out_dir, resolution):

    # load json data from the selected pair
    data = []
    for p in [json_path_l, json_path_r]:
        with open(p) as f:
            data.append(json.load(f))

    # create s2p config
    config = {"images": [{"img": os.path.join(img_dir, data[0]["img"]), "rpc": data[0]["rpc"]},
                         {"img": os.path.join(img_dir, data[1]["img"]), "rpc": data[1]["rpc"]}],
              "out_dir": ".",
              "roi": {"x": 0, "y": 0, "w": data[0]["width"], "h": data[0]["height"]},
              "dsm_resolution": resolution}
    # sanity check
    for i in [0, 1]:
        if not os.path.exists(config["images"][i]["img"]):
            raise FileNotFoundError("Could not find {}".format(config["images"][i]["img"]))

    # write s2p config to disk
    img_id_l = os.path.splitext(os.path.basename(json_path_l))[0]
    img_id_r = os.path.splitext(os.path.basename(json_path_r))[0]
    s2p_out_dir = os.path.join(out_dir, "{}_{}".format(img_id_l, img_id_r))
    os.makedirs(s2p_out_dir, exist_ok=True)
    config_path = os.path.join(s2p_out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # run s2p and redirect output to log file
    log_file = os.path.join(s2p_out_dir, 'log.txt')
    with open(log_file, 'w') as outfile:
        subprocess.run(['s2p', config_path], stdout=outfile, stderr=outfile)

def load_heuristic_pairs(root_dir, img_dir, heuristic_pairs_file, n_pairs=1):

    # link msi ids to rgb geotiff ids
    img_paths = glob.glob(os.path.join(img_dir, "*.tif"))
    msi_id_to_rgb_id ={}
    for p in img_paths:
        rgb_id = os.path.splitext(os.path.basename(p))[0]
        with rasterio.open(p, "r") as f:
            msi_id = f.tags()["NITF_IID2"].split("-")[0]
            msi_id_to_rgb_id[msi_id] = rgb_id

    selected_pairs_json_paths = []
    with open(heuristic_pairs_file, 'r') as f:
        lines = f.read().split("\n")
    n_selected = 0
    for l in lines:
        tmp = l.split(" ")
        msi_id_l, msi_id_r = os.path.basename(tmp[0]).split("-")[0], os.path.basename(tmp[1]).split("-")[0]
        if msi_id_l in msi_id_to_rgb_id.keys() and msi_id_r in msi_id_to_rgb_id.keys():
            json_path_l = os.path.join(root_dir, "{}.json".format(msi_id_to_rgb_id[msi_id_l]))
            json_path_r = os.path.join(root_dir, "{}.json".format(msi_id_to_rgb_id[msi_id_r]))
            selected_pairs_json_paths.append((json_path_l, json_path_r))
            n_selected += 1
        if n_selected >= n_pairs:
            break

    return selected_pairs_json_paths

def eval_s2p(aoi_id, root_dir, dfc_dir, output_dir="s2p_dsms", resolution=0.5, n_pairs=1):

    out_dir = os.path.join(output_dir, aoi_id)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    img_dir = os.path.join(dfc_dir, "Track3-RGB/{}".format(aoi_id))

    heuristic = True
    heuristic_pairs_file = os.path.join(dfc_dir, "DFC2019_JAX_heuristic_pairs.txt")
    if heuristic and os.path.exists(heuristic_pairs_file):
        selected_pairs_json_paths = load_heuristic_pairs(root_dir, img_dir, heuristic_pairs_file)
        print("{} heuristic pairs selected".format(n_pairs))
    else:
        selected_pairs_json_paths, n_possible_pairs = select_pairs(root_dir, n_pairs=n_pairs)
        print("{} random pairs selected from {} possible".format(n_pairs, n_possible_pairs))

    for t, (json_path_l, json_path_r) in enumerate(selected_pairs_json_paths):
        print("Running s2p ! Pair {} of {}...".format(t+1, n_pairs))
        run_s2p(json_path_l, json_path_r, img_dir, out_dir, resolution)
        print("...done")
    s2p_ply_paths = glob.glob(os.path.join(out_dir, "*/*/*/*/cloud.ply"))
    shutil.rmtree("s2p_tmp")

    # merge s2p pairwise dsms
    from plyflatten import plyflatten_from_plyfiles_list
    raster, profile = plyflatten_from_plyfiles_list(s2p_ply_paths, resolution=resolution, radius=3)
    profile["dtype"] = raster.dtype
    profile["height"] = raster.shape[0]
    profile["width"] = raster.shape[1]
    profile["count"] = 1
    profile["driver"] = "GTiff"
    mvs_dsm_path = os.path.join(out_dir, "mvs_dsm_{}_pairs.tif".format(n_pairs))
    with rasterio.open(mvs_dsm_path, "w", **profile) as f:
        f.write(raster[:, :, 0], 1)

    # evaluate s2p generated mvs DSM
    gt_dsm_path = os.path.join(dfc_dir, "Track3-Truth/{}_DSM.tif".format(aoi_id))
    gt_roi_metadata = np.loadtxt(os.path.join(dfc_dir, "Track3-Truth/{}_DSM.txt".format(aoi_id)))
    gt_seg_path = os.path.join(dfc_dir, "Track3-Truth/{}_CLS.tif".format(aoi_id))
    abs_err = dsm_pointwise_abs_errors(mvs_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path)
    print("Path to output S2P MVS DSM: {}".format(mvs_dsm_path))
    print("Altitude MAE: {}".format(np.nanmean(abs_err)))

if __name__ == '__main__':
    import fire
    fire.Fire(eval_s2p)
