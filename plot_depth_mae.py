import matplotlib.pyplot as plt
import numpy as np
import rasterio
import os
import glob

##################
# inputs
##################

gt_dir = "../Datasets/DFC2019/Track3-Truth"
logs_dir = "logs"
aoi_id = "JAX_004"

"""
exp_names = ["2021-06-08_23:04:28_snerf_siren_h256_test"]
"""

#exp_names = ["2021-06-02_14:41:41_nerf_classic_fine128",
#             "2021-06-08_11:52:10_snerf_pe_h256",
#             "2021-06-08_16:37:35_snerf_siren_h256"]

exp_names = ["2021-06-16_04:58:08_nerf_jax004",
             "2021-06-16_04:50:03_snerf_jax004"]


#####################
# script starts here
#####################

def sorted_nicely(l):
    import re
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def compute_absolute_error(dsm_path, gt_path, err_path=None):
    # read gt dsm
    with rasterio.open(gt_path, "r") as f:
        gt = f.read()[0, :, :]
    # read predicted dsm
    with rasterio.open(dsm_path, "r") as f:
        profile = f.profile
        dsm = f.read()[0, :, :]
    err = np.absolute(gt - dsm)
    if err_path is not None:
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with rasterio.open(err_path, "w", **profile) as f:
            f.write(err, 1)
    return err

# only if you are checking old runs (previous to June 8)
old_outputs = False

# the lists with the evolution of the mae for each run will be stored in a dictionary
mae = {}
for e in exp_names:
    mae[e] = []

gt_path = os.path.join(gt_dir, "{}_DSM.tif".format(aoi_id))

if old_outputs:
    for e in exp_names:
        log_path = os.path.join(logs_dir, e)
        dsm_paths = sorted_nicely(glob.glob(os.path.join(log_path, "val/dsm/*.tif")))
        for d in dsm_paths:
            err_path = d.replace("/dsm/", "/dsm_err/")
            err = compute_absolute_error(d, gt_path, err_path)
            mae[e].append(np.nanmean(err.ravel()))
else:
    roi_txt = os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))
    gt_roi_metadata = np.loadtxt(roi_txt)
    xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
    xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
    resolution = gt_roi_metadata[3]
    # projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    for e in exp_names:
        log_path = os.path.join(logs_dir, e)
        dsm_paths = sorted_nicely(glob.glob(os.path.join(log_path, "val/dsm/*.tif")))
        print(dsm_paths)
        for d in dsm_paths:
            crop_path = d.replace("/dsm/", "/dsm_crops/")
            err_path = d.replace("/dsm/", "/dsm_err/")
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            # TODO maybe rasterio can do this faster ?
            if not os.path.exists(crop_path):
                os.system("gdal_translate -projwin {} {} {} {} {} {}".format(ulx, uly, lrx, lry, d, crop_path))
            err = compute_absolute_error(crop_path, gt_path, err_path)
            mae[e].append(np.nanmean(err.ravel()))


colors = ["blue", "firebrick", "magenta", "green", "grey", "darkorange"]

plt.figure(figsize=(10, 5))
labels = list(mae.keys())
for i, (e, c) in enumerate(zip(exp_names, colors)):
    plt.plot(np.arange(1, len(mae[e])+1), mae[e], color=c)
    labels[i] += "    last: {:.2f} m".format(mae[e][-1])
plt.legend(labels)
plt.savefig('mae.png', bbox_inches='tight')










