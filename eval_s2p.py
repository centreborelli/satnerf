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


def geojson_polygon(coords_array):
    """
    define a geojson polygon from a Nx2 numpy array with N 2d coordinates delimiting a boundary
    """
    from shapely.geometry import Polygon

    # first attempt to construct the polygon, assuming the input coords_array are ordered
    # the centroid is computed using shapely.geometry.Polygon.centroid
    # taking the mean is easier but does not handle different densities of points in the edges
    pp = coords_array.tolist()
    poly = Polygon(pp)
    x_c, y_c = np.array(poly.centroid.xy).ravel()

    # check that the polygon is valid, i.e. that non of its segments intersect
    # if the polygon is not valid, then coords_array was not ordered and we have to do it
    # a possible fix is to sort points by polar angle using the centroid (anti-clockwise order)
    if not poly.is_valid:
        pp.sort(key=lambda p: np.arctan2(p[0] - x_c, p[1] - y_c))

    # construct the geojson
    geojson_polygon = {"coordinates": [pp], "type": "Polygon"}
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon

def lonlat_from_utm(easts, norths, zonestring):
    """
    convert utm to lon-lat
    """
    import pyproj
    proj_src = pyproj.Proj("+proj=utm +zone=%s" % zonestring)
    proj_dst = pyproj.Proj("+proj=latlong")
    return pyproj.transform(proj_src, proj_dst, easts, norths)

def read_DFC2019_lonlat_aoi(aoi_id, dfc_dir):
    if aoi_id[:3] == "JAX":
        zonestring = "17R"
    else:
        raise ValueError("AOI not valid. Expected JAX_(3digits) but received {}".format(aoi_id))
    roi = np.loadtxt(os.path.join(dfc_dir, "Track3-Truth/" + aoi_id + "_DSM.txt"))
    xoff, yoff, xsize, ysize, resolution = roi[0], roi[1], int(roi[2]), int(roi[2]), roi[3]
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff
    xmin, xmax, ymin, ymax = ulx, lrx, uly, lry
    easts = [xmin, xmin, xmax, xmax, xmin]
    norths = [ymin, ymax, ymax, ymin, ymin]
    lons, lats = lonlat_from_utm(easts, norths, zonestring)
    lonlat_bbx = geojson_polygon(np.vstack((lons, lats)).T)
    return lonlat_bbx

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
        tmp_gt_path = os.path.join(os.path.dirname(out_rdsm_path), "tmp_gt_{}.tif".format(unique_identifier))
        with rasterio.open(gt_dsm_path, "r") as f:
            gt_dsm = f.read()[0, :, :]
        with rasterio.open(tmp_gt_path, 'w', **profile) as dst:
            gt_dsm[water_mask.astype(bool)] = np.nan
            dst.write(gt_dsm, 1)
        gt_dsm_path = tmp_gt_path

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
        kwds = f.profile
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
    abs_err = pred_rdsm - gt_dsm

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

    return abs(abs_err)

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


def run_s2p(json_path_l, json_path_r, img_dir, out_dir, resolution, prefix="", aoi=None):

    # load json data from the selected pair
    data = []
    for p in [json_path_l, json_path_r]:
        with open(p) as f:
            data.append(json.load(f))

    # create s2p config
    use_pan = True
    if use_pan:
        aoi_id = data[0]["img"][:7]
        if aoi_id in ["JAX_004", "JAX_068"]:
            pan_dir = "/vsicurl/http://138.231.80.166:2332/grss-2019/track_3/Track3-MSI-1/"
        else:
            pan_dir = "/vsicurl/http://138.231.80.166:2332/grss-2019/track_3/Track3-MSI-3/"
        img_path1 = pan_dir + data[0]["img"].replace("RGB", "PAN")
        img_path2 = pan_dir + data[1]["img"].replace("RGB", "PAN")
    else:
        img_path1 = os.path.join(img_dir, data[0]["img"])
        img_path2 = os.path.join(img_dir, data[1]["img"])
    config = {"images": [{"img": img_path1, "rpc": data[0]["rpc"]},
                         {"img": img_path2, "rpc": data[1]["rpc"]}],
              "out_dir": ".",
              "dsm_resolution": resolution,
              "rectification_method": "sift",
              "matching_algorithm": "mgm_multi"}
    if aoi is None:
        config["roi"] = {"x": 0, "y": 0, "w": data[0]["width"], "h": data[0]["height"]}
    else:
        config["roi_geojson"] = aoi

    # sanity check
    if not use_pan:
        for i in [0, 1]:
            if not os.path.exists(config["images"][i]["img"]):
                raise FileNotFoundError("Could not find {}".format(config["images"][i]["img"]))

    # write s2p config to disk
    img_id_l = os.path.splitext(os.path.basename(json_path_l))[0]
    img_id_r = os.path.splitext(os.path.basename(json_path_r))[0]
    s2p_out_dir = os.path.join(out_dir, "{}{}_{}".format(prefix, img_id_l, img_id_r))
    os.makedirs(s2p_out_dir, exist_ok=True)
    config_path = os.path.join(s2p_out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # run s2p and redirect output to log file
    log_file = os.path.join(s2p_out_dir, 'log.txt')
    if not os.path.exists(os.path.join(s2p_out_dir, 'dsm.tif')):
        with open(log_file, 'w') as outfile:
            subprocess.run(['s2p', config_path], stdout=outfile, stderr=outfile)

def load_heuristic_pairs(root_dir, img_dir, heuristic_pairs_file, n_pairs=1):

    # link msi ids to rgb geotiff ids
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
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

def project_cloud_into_utm_grid(xyz, bb, definition, mode, mask=None):
    # possible modes = 'min', 'max', 'avg', 'med'
    from itertools import groupby

    origin = np.array([bb[0], bb[2]])
    w, h = bb[1] - bb[0], bb[3] - bb[2]
    map_w = int(round(w / definition)) + 1
    map_h = int(round(h / definition)) + 1

    map_np = np.zeros((map_h, map_w), dtype=float)
    map_np[:,:] = np.nan
    coords = np.round((xyz[:,:2] - origin) / definition).astype(int)
    
    # sanity check
    valid_rows = np.logical_and(coords[:,1] < map_h, coords[:,1] >= 0)
    valid_cols = np.logical_and(coords[:,0] < map_w, coords[:,0] >= 0)
    valid_coords_indices = np.logical_and(valid_rows, valid_cols)
    coords = coords[valid_coords_indices, :]
    xyz = xyz[valid_coords_indices, :]
    
    if mask is None:
        mask = np.zeros((map_h, map_w), dtype=int)

    if mode == 'min' or mode == 'max':
        if mode == 'min':
            idx = np.flip(np.argsort(xyz[:,2]))
        else:
            idx = np.argsort(xyz[:,2])   
        coords, data_np = coords[idx], xyz[idx]
        map_np[coords[:,1], coords[:,0]] = data_np[:,2]
    else:
        coords_unique, coords_indices = np.unique(coords, return_inverse=True, axis=0)
        sorted_id_z = sorted(list(zip(coords_indices, xyz[:,2])), key=lambda x: x[0])
        groups_id_z = groupby(sorted_id_z, lambda x: x[0])
        
        dsm_z = []
        if mode == 'avg':
            #dsm_heights = [np.mean(cloud_heights[coords_indices == i]) for i in np.arange(coords_unique.shape[0])]
            dsm_z = [np.mean(np.array(list(g))[:,1]) for k, g in groups_id_z]
        else:
            #dsm_z = [np.median(cloud_heights[coords_indices == i]) for i in np.arange(coords_unique.shape[0])] # (~180s/dsm)
            dsm_z = [np.median(np.array(list(g))[:,1]) for k, g in groups_id_z] #(~10s/dsm)
            
        map_np[coords_unique[:,1], coords_unique[:,0]] = np.array(dsm_z)
             
    if (np.sum(np.logical_not(np.isnan(map_np))) < 3):
        print ('There are less than 3 points.')
    
    raw_map_np = map_np.copy()
    raw_map_np = np.flipud(raw_map_np)
    
    return raw_map_np

def eval_s2p(aoi_id, root_dir, dfc_dir, output_dir="s2p_dsms", resolution=0.5, n_pairs=1, crops=True):

    if crops:
        print("using crops")
        img_dir = os.path.join(dfc_dir, "Track3-RGB-crops/{}".format(aoi_id))
        output_dir += "_crops"
    else:
        img_dir = os.path.join(dfc_dir, "Track3-RGB/{}".format(aoi_id))

    out_dir = os.path.join(output_dir, aoi_id)
    #if os.path.exists(out_dir):
    #    shutil.rmtree(out_dir)

    heuristic = True
    heuristic_pairs_file = os.path.join(dfc_dir, "DFC2019_JAX_heuristic_pairs.txt")
    if heuristic and os.path.exists(heuristic_pairs_file):
        selected_pairs_json_paths = load_heuristic_pairs(root_dir, img_dir, heuristic_pairs_file, n_pairs=n_pairs)
        print("{} heuristic pairs selected".format(n_pairs))
    else:
        selected_pairs_json_paths, n_possible_pairs = select_pairs(root_dir, n_pairs=n_pairs)
        print("{} random pairs selected from {} possible".format(n_pairs, n_possible_pairs))

    lonlat_aoi = read_DFC2019_lonlat_aoi(aoi_id, dfc_dir)

    for t, (json_path_l, json_path_r) in enumerate(selected_pairs_json_paths):
        print("Running s2p ! Pair {} of {}...".format(t+1, n_pairs))
        run_s2p(json_path_l, json_path_r, img_dir, out_dir, resolution, aoi=lonlat_aoi, prefix="{:02}_".format(t))
        print("...done")
    s2p_ply_paths = glob.glob(os.path.join(out_dir, "*/*/*/*/cloud.ply"))
    shutil.rmtree("s2p_tmp")

    # merge s2p pairwise dsms (mean)
    from plyflatten import plyflatten_from_plyfiles_list
    raster, profile = plyflatten_from_plyfiles_list(s2p_ply_paths, resolution=resolution, radius=2)
    profile["dtype"] = raster.dtype
    profile["height"] = raster.shape[0]
    profile["width"] = raster.shape[1]
    profile["count"] = 1
    profile["driver"] = "GTiff"
    mvs_dsm_path = os.path.join(out_dir, "mvs_dsm_{}_pairs_avg.tif".format(n_pairs))
    with rasterio.open(mvs_dsm_path, "w", **profile) as f:
        f.write(raster[:, :, 0], 1)
    # evaluate s2p generated mvs DSM
    gt_dsm_path = os.path.join(dfc_dir, "Track3-Truth/{}_DSM.tif".format(aoi_id))
    gt_roi_metadata = np.loadtxt(os.path.join(dfc_dir, "Track3-Truth/{}_DSM.txt".format(aoi_id)))
    if aoi_id in ["JAX_004", "JAX_260"]:
        gt_seg_path = os.path.join(dfc_dir, "Track3-Truth/{}_CLS_v2.tif".format(aoi_id))
    else:
        gt_seg_path = os.path.join(dfc_dir, "Track3-Truth/{}_CLS.tif".format(aoi_id))
    rmvs_dsm_path = os.path.join(out_dir, "rmvs_avg_dsm_{}_pairs.tif".format(n_pairs))
    abs_err = dsm_pointwise_abs_errors(mvs_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path, out_rdsm_path=rmvs_dsm_path)
    print("Path to output S2P MVS DSM: {}".format(mvs_dsm_path))
    print("Altitude MAE: {}".format(np.nanmean(abs_err)))
    shutil.copyfile(rmvs_dsm_path, rmvs_dsm_path.replace(".tif", "_{:.3f}.tif".format(np.nanmean(abs_err))))
    with rasterio.open(rmvs_dsm_path, "r") as f:
        avg_dsm = f.read(1)
        profile = f.profile
    os.remove(rmvs_dsm_path)

    # merge s2p pairwise dsms (median)
    from s2p import ply
    xyz = np.vstack([ply.read_3d_point_cloud_from_ply(p)[0][:,:3] for p in s2p_ply_paths])
    # read dsm metadata
    dsm_metadata = np.loadtxt(os.path.join(dfc_dir, "Track3-Truth/{}_DSM.txt".format(aoi_id)))
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution_ = dsm_metadata[3]
    # define projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution_, xoff + xsize * resolution_, yoff
    bb = [ulx, lrx, lry, uly]
    #bb = [np.min(xyz[:, 0]), np.max(xyz[:, 0]), np.min(xyz[:, 1]), np.max(xyz[:, 1])]
    med_dsm = project_cloud_into_utm_grid(xyz, bb, resolution, 'med', mask=None)
    profile["height"] = med_dsm.shape[0]
    profile["width"] = med_dsm.shape[1]
    mvs_dsm_path = os.path.join(out_dir, "mvs_dsm_{}_pairs_med.tif".format(n_pairs))
    with rasterio.open(mvs_dsm_path, "w", **profile) as f:
        f.write(med_dsm, 1)
    # evaluate s2p generated mvs DSM
    gt_dsm_path = os.path.join(dfc_dir, "Track3-Truth/{}_DSM.tif".format(aoi_id))
    gt_roi_metadata = np.loadtxt(os.path.join(dfc_dir, "Track3-Truth/{}_DSM.txt".format(aoi_id)))
    if aoi_id in ["JAX_004", "JAX_260"]:
        gt_seg_path = os.path.join(dfc_dir, "Track3-Truth/{}_CLS_v2.tif".format(aoi_id))
    else:
        gt_seg_path = os.path.join(dfc_dir, "Track3-Truth/{}_CLS.tif".format(aoi_id))
    rmvs_dsm_path = os.path.join(out_dir, "rmvs_med_dsm_{}_pairs.tif".format(n_pairs))
    rmvs_err_path = os.path.join(out_dir, "rmvs_med_err_{}_pairs.tif".format(n_pairs))
    abs_err = dsm_pointwise_abs_errors(mvs_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path, out_rdsm_path=rmvs_dsm_path, out_err_path=rmvs_err_path)
    print("Altitude MAE: {}".format(np.nanmean(abs_err)))
    with rasterio.open(rmvs_dsm_path, "r") as f:
        med_dsm = f.read(1)
    med_nans = np.isnan(med_dsm)
    med_dsm[med_nans] = avg_dsm[med_nans]
    with rasterio.open(rmvs_dsm_path.replace(".tif", "_{:.3f}.tif".format(np.nanmean(abs_err))), "w+", **profile) as f:
        f.write(med_dsm, 1)
    os.remove(rmvs_dsm_path)
    
    


if __name__ == '__main__':
    import fire
    fire.Fire(eval_s2p)
