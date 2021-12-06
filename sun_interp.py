import glob
import numpy as np
import os
import json
import rpcm
from eval_aoi import load_nerf, batched_inference, save_nerf_output_to_images, find_best_embbeding_for_val_image
import torch
import shutil
from datasets import SatelliteDataset

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")



def sun_interp(run_id, logs_dir, output_dir, epoch_number, checkpoints_dir=None):

    log_path = os.path.join(logs_dir, run_id)
    with open('{}/opts.json'.format(log_path), 'r') as f:
            args = json.load(f)

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

    # uncomment lines below to save all images sorted in chronological order or in order of solar incidence
    """
    json_paths_ordered = np.array(json_paths)[np.argsort(dates)].tolist()
    for p, i in zip(json_paths_ordered, np.sort(dates)):
        out_p = os.path.join(output_dir, "sorted_by_date/{:06}_{}".format(i, os.path.basename(p).replace(".json", ".tif")))
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        shutil.copy(os.path.join(args["img_dir"], os.path.basename(p).replace(".json", ".tif")), out_p)
    json_paths_ordered = np.array(json_paths)[np.argsort(solar_incidence_angles)].tolist()
    for p, i in zip(json_paths_ordered, np.sort(solar_incidence_angles)):
        out_p = os.path.join(output_dir, "sorted_by_sun/{:.2f}_{}".format(i, os.path.basename(p).replace(".json", ".tif")))
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        shutil.copy(os.path.join(args["img_dir"], os.path.basename(p).replace(".json", ".tif")), out_p)
    ie += 1
    """

    # take image closest to nadir as reference view
    reference_image = json_paths[np.argmin(incidence_angles)]
    # define solar direction bounds
    upper_sun_dir = solar_dir_vectors[np.argmin(solar_incidence_angles)] # sun is close to nadir
    lower_sun_dir = solar_dir_vectors[np.argmax(solar_incidence_angles)] # sun is very tilted

    # define a sat-nerf dataset of one single image using the reference view
    dataset = SatelliteDataset(args["root_dir"], args["img_dir"], split="val",
                               img_downscale=args["img_downscale"], cache_dir=args["cache_dir"])
    dataset.json_files = [reference_image]

    # load nerf
    if checkpoints_dir is None:
        checkpoints_dir = args["checkpoints_dir"]
    models, conf = load_nerf(run_id, log_path, checkpoints_dir, epoch_number-1)

    # select ts if s-nerf-w
    if conf.name == "s-nerf-w":
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

    # run nerf for a range of vectors interpolated between solar direction bounds
    n_interp = 5
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
        results = batched_inference(models, rays.cuda(), ts, conf)

        # save results
        for k in sample.keys():
            if torch.is_tensor(sample[k]):
                sample[k] = sample[k].unsqueeze(0)
            else:
                sample[k] = [sample[k]]
        out_dir = os.path.join(output_dir, "sun_interp", run_id)
        os.makedirs(out_dir, exist_ok=True)
        save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number)
        output_im_paths = glob.glob(os.path.join(out_dir, "*/*epoch{}.tif".format(epoch_number)))
        for p in output_im_paths:
            shutil.move(p, p.replace(".tif", "_solar_incidence_angle_{:.2f}deg.tif".format(solar_incidence_angle)))
        print("solar incidence angle {:.2f} completed ({} of {})".format(solar_incidence_angle, i+1, n_interp))

if __name__ == '__main__':
    import fire
    fire.Fire(sun_interp)