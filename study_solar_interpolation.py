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

from PIL import Image
import cv2
import rpcm


import warnings
warnings.filterwarnings("ignore")


def hstack_sun_tifs(img_paths, crop=True):
    images = []
    for p in img_paths:
        with rasterio.open(p) as f:
            img = f.read()
        img = img.transpose(1, 2, 0)
        if crop:
            h, w = img.shape[:2]
            row_start, row_end = int(h/4), int(3*h/4)
            col_start, col_end = int(w/4), int(3*w/4)
            img = img[row_start:row_end, col_start:col_end]
        images.append(img)
    img = np.hstack(images)[:, :, 0]
    return (img*255).astype(np.uint8) #np.dstack([img, img, img])

def hstack_rgb_tifs(img_paths, crop=True):
    images = []
    for p in img_paths:
        with rasterio.open(p) as f:
            img = f.read()
        img = img.transpose(1, 2, 0)
        if crop:
            h, w = img.shape[:2]
            row_start, row_end = int(h/4), int(3*h/4)
            col_start, col_end = int(w/4), int(3*w/4)
            img = img[row_start:row_end, col_start:col_end, :]
        images.append(img)
    img = np.hstack(images)
    return (img*255).astype(np.uint8)

def quickly_interpolate_nans_from_singlechannel_img(image, method='nearest'):
    from scipy import interpolate
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    mask = np.isnan(image.reshape(h, w))
    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]
    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y), method=method
    )
    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values
    return interp_image

def hstack_dsm_tifs_v1(img_paths, cmap=cv2.COLORMAP_VIRIDIS, crop=True, vmax=None, vmin=None):
    images = []
    for p in img_paths:
        with rasterio.open(p) as f:
            img = f.read()
        img = img.transpose(1, 2, 0)
        if crop:
            h, w = img.shape[:2]
            row_start, row_end = int(h/4), int(3*h/4)
            col_start, col_end = int(w/4), int(3*w/4)
            img = img[row_start:row_end, col_start:col_end, 0]
        x = img
        from scipy import interpolate
        #x = np.nan_to_num(x) # change nan to 0
        x = quickly_interpolate_nans_from_singlechannel_img(x)
        mi = np.min(x) if vmin is None else vmin
        ma = np.max(x) if vmax is None else vmax
        x = np.clip(x, mi, ma)
        x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
        x = (255*x).astype(np.uint8)
        x = np.clip(x, 0, 255)
        x_ = cv2.applyColorMap(x, cmap)
        x_ = cv2.cvtColor(x_, cv2.COLOR_BGR2RGB)
        images.append(x_)
    img = np.hstack(images)
    return img


def hstack_dsm_tifs_v2(img_paths, crop=True):
    import sys
    sys.path.append('/home/roger/demtk')
    import iio, demtk
    images = []
    for p in img_paths:
        with rasterio.open(p) as f:
            img = f.read()
        img = img.transpose(1, 2, 0)[:, :, 0]
        if crop:
            h, w = img.shape[:2]
            row_start, row_end = int(h/4), int(3*h/4)
            col_start, col_end = int(w/4), int(3*w/4)
            img = img[row_start:row_end, col_start:col_end]
        img = demtk.renderclean(img)
        images.append(img)
    img = np.hstack(images)
    return img

def sun_interp(run_id, logs_dir, output_dir, epoch_number, checkpoints_dir=None, root_dir=None, img_dir=None, gt_dir=None):

    print(logs_dir)
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

    # load pretrained nerf
    if checkpoints_dir is None:
        checkpoints_dir = args.ckpts_dir
    models = load_nerf(run_id, logs_dir, checkpoints_dir, epoch_number-1)

    json_paths = glob.glob(os.path.join(args.root_dir, "*.json"))
    solar_incidence_angles, solar_dir_vectors = [], []
    for json_p in json_paths:
        # read json
        with open(json_p) as f:
            d = json.load(f)
        # get solar direction vectors and solar incidence angle
        sun_el = np.radians(float(d["sun_elevation"]))
        sun_az = np.radians(float(d["sun_azimuth"]))
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        surface_normal = np.array([0., 0., 1.0])
        u1 = sun_d / np.linalg.norm(sun_d)
        u2 = surface_normal / np.linalg.norm(surface_normal)
        solar_dir_vectors.append(sun_d)
        solar_incidence_angles.append(np.degrees(np.arccos(np.dot(u1, u2))))

    # take image closest to nadir as reference view
    reference_image = sat_utils.sort_by_increasing_view_incidence_angle(args.root_dir)[0]
    # define solar direction bounds
    upper_sun_dir = solar_dir_vectors[np.argmin(solar_incidence_angles)] # sun is close to nadir
    lower_sun_dir = solar_dir_vectors[np.argmax(solar_incidence_angles)] # sun is very tilted

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

    out_dir = os.path.join(output_dir, run_id)

    # run nerf for a range of vectors interpolated between solar direction bounds
    n_interp = 10
    for i, alpha in enumerate(np.linspace(0, 1, n_interp)):

        # define current solar incidence angle
        sun_d = alpha * upper_sun_dir + (1 - alpha) * lower_sun_dir
        surface_normal = np.array([0., 0., 1.0])
        u1 = sun_d / np.linalg.norm(sun_d)
        u2 = surface_normal / np.linalg.norm(surface_normal)
        solar_incidence_angle = np.degrees(np.arccos(np.dot(u1, u2)))

        # define nerf input and run the model
        sample = dataset[0]
        rays = sample["rays"]
        sun_dirs = torch.from_numpy(np.tile(sun_d, (rays.shape[0], 1)))
        rays[:, 8:11] = sun_dirs.type(torch.FloatTensor)
        results = batched_inference(models, rays.cuda(), ts, args)

        # save results
        for k in sample.keys():
            if torch.is_tensor(sample[k]):
                sample[k] = sample[k].unsqueeze(0)
            else:
                sample[k] = [sample[k]]

        os.makedirs(out_dir, exist_ok=True)
        save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number)
        output_im_paths = glob.glob(os.path.join(out_dir, "*/*epoch{}.tif".format(epoch_number)))
        for p in output_im_paths:
            shutil.move(p, p.replace(".tif", "_solar_incidence_angle_{:.2f}deg.tif".format(solar_incidence_angle)))
        print("solar incidence angle {:.2f} completed ({} of {})".format(solar_incidence_angle, i+1, n_interp))

    crop_summary_images = True

    # write summary images
    summary_dir = os.path.join(out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    # sun
    img_paths = sorted(glob.glob(os.path.join(out_dir, "sun/*.tif")))
    out_img = Image.fromarray(hstack_sun_tifs(img_paths, crop=crop_summary_images))
    out_img.save(os.path.join(summary_dir, "sun.png"))
    # albedo
    img_paths = sorted(glob.glob(os.path.join(out_dir, "albedo/*.tif")))
    out_img = Image.fromarray(hstack_rgb_tifs(img_paths, crop=crop_summary_images))
    out_img.save(os.path.join(summary_dir, "albedo.png"))
    # rgbs
    img_paths = sorted(glob.glob(os.path.join(out_dir, "rgb/*.tif")))
    out_img = Image.fromarray(hstack_rgb_tifs(img_paths, crop=crop_summary_images))
    out_img.save(os.path.join(summary_dir, "rgb.png"))
    # depth v1
    img_paths = sorted(glob.glob(os.path.join(out_dir, "depth/*.tif")))
    out_img = Image.fromarray(hstack_dsm_tifs_v1(img_paths, crop=crop_summary_images))
    out_img.save(os.path.join(summary_dir, "depth_v1.png"))
    # depth v2
    try:
        img_paths = sorted(glob.glob(os.path.join(out_dir, "depth/*.tif")))
        out_img = Image.fromarray(hstack_dsm_tifs_v2(img_paths, crop=crop_summary_images))
        out_img.save(os.path.join(summary_dir, "depth_v2.png"))
    except:
        print("warning: dmtk shading failed")

if __name__ == '__main__':
    import fire
    fire.Fire(sun_interp)
