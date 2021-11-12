import rpcm
import glob
from bundle_adjust.cam_utils import SatelliteImage
from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline
from bundle_adjust import loader
import os
import numpy as np
import srtm4
import shutil


def rio_open(*args,**kwargs):
    import rasterio
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return rasterio.open(*args,**kwargs)

def get_image_lonlat_aoi(rpc, h, w):
    z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
    cols, rows, alts = [0,w,w,0], [0,0,h,h], [z]*4
    lons, lats = rpc.localization(cols, rows, alts)
    lonlat_coords = np.vstack((lons, lats)).T
    geojson_polygon = {"coordinates": [lonlat_coords.tolist()], "type": "Polygon"}
    x_c = lons.min() + (lons.max() - lons.min())/2
    y_c = lats.min() + (lats.max() - lats.min())/2
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon

def run_ba(aoi_id, dfc_dir, output_dir):

    # load input data
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(dfc_dir, "Track3-RGB/{}".format(aoi_id))
    myimages = sorted(glob.glob(img_dir + "/*.tif"))
    myrpcs = [rpcm.rpc_from_geotiff(p) for p in myimages]
    input_images = [SatelliteImage(fn, rpc) for fn, rpc in zip(myimages, myrpcs)]
    ba_input_data = {}
    ba_input_data['in_dir'] = img_dir
    ba_input_data['out_dir'] = os.path.join(output_dir, "ba_files")
    ba_input_data['images'] = input_images
    print('Input data set!\n')

    # run bundle adjustment
    tracks_config = {'FT_reset': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based', "FT_K": 300}
    ba_extra = {"cam_model": "rpc"}
    ba_pipeline = BundleAdjustmentPipeline(ba_input_data, tracks_config=tracks_config, extra_ba_config=ba_extra)
    ba_pipeline.run()

    # save all bundle adjustment parameters in a temporary directory
    ba_params_dir = os.path.join(ba_pipeline.out_dir, "ba_params")
    os.makedirs(ba_params_dir, exist_ok=True)
    np.save(os.path.join(ba_params_dir, "pts_ind.npy"), ba_pipeline.ba_params.pts_ind)
    np.save(os.path.join(ba_params_dir, "cam_ind.npy"), ba_pipeline.ba_params.cam_ind)
    np.save(os.path.join(ba_params_dir, "pts3d.npy"), ba_pipeline.ba_params.pts3d_ba - ba_pipeline.global_transform)
    np.save(os.path.join(ba_params_dir, "pts2d.npy"), ba_pipeline.ba_params.pts2d)
    fnames_in_use = [ba_pipeline.images[idx].geotiff_path for idx in ba_pipeline.ba_params.cam_prev_indices]
    loader.save_list_of_paths(os.path.join(ba_params_dir, "geotiff_paths.txt"), fnames_in_use)

def create_dataset_from_DFC2019_data(aoi_id, dfc_dir, output_dir, use_ba=False):

    # create a json file of metadata for each input image
    # contains: h, w, rpc, sun elevation, sun azimuth, acquisition date
    #           + geojson polygon with the aoi of the image
    os.makedirs(output_dir, exist_ok=True)
    path_to_dsm = os.path.join(dfc_dir, "Track3-Truth/{}_DSM.tif".format(aoi_id))
    if aoi_id[:3] == "JAX":
        path_to_msi = "http://138.231.80.166:2334/core3d/Jacksonville/WV3/MSI"
    elif aoi_id[:3] == "OMA":
        path_to_msi = "http://138.231.80.166:2334/core3d/Omaha/WV3/MSI"
    if use_ba:
        geotiff_paths = loader.load_list_of_paths(os.path.join(output_dir, "ba_files/ba_params/geotiff_paths.txt"))
        ba_geotiff_basenames = [os.path.basename(x) for x in geotiff_paths]
        ba_kps_pts3d_ind = np.load(os.path.join(output_dir, "ba_files/ba_params/pts_ind.npy"))
        ba_kps_cam_ind = np.load(os.path.join(output_dir, "ba_files/ba_params/cam_ind.npy"))
        ba_kps_pts2d = np.load(os.path.join(output_dir, "ba_files/ba_params/pts2d.npy"))
    else:
        img_dir = os.path.join(dfc_dir, "Track3-RGB/{}".format(aoi_id))
        geotiff_paths = sorted(glob.glob(img_dir + "/*.tif"))

    for rgb_p in geotiff_paths:
        d = {}
        d["img"] = os.path.basename(rgb_p)

        src = rio_open(rgb_p)
        d["height"] = int(src.meta["height"])
        d["width"] = int(src.meta["width"])
        original_rpc = rpcm.RPCModel(src.tags(ns='RPC'), dict_format="geotiff")

        img_id = src.tags()["NITF_IID2"].replace(" ", "_")
        msi_p = "{}/{}.NTF".format(path_to_msi, img_id)
        src = rio_open(msi_p)
        d["sun_elevation"] = src.tags()["NITF_USE00A_SUN_EL"]
        d["sun_azimuth"] = src.tags()["NITF_USE00A_SUN_AZ"]
        d["acquisition_date"] = src.tags()['NITF_STDIDC_ACQUISITION_DATE']
        d["geojson"] = get_image_lonlat_aoi(original_rpc, d["height"], d["width"])

        src = rio_open(path_to_dsm)
        dsm = src.read()[0, :, :]
        d["min_alt"] = int(np.round(dsm.min() - 10))
        d["max_alt"] = int(np.round(dsm.max() + 10))

        if use_ba:
            # use corrected rpc
            rpc_path = os.path.join(output_dir, "ba_files/rpcs_adj/{}.rpc_adj".format(loader.get_id(rgb_p)))
            d["rpc"] = rpcm.rpc_from_rpc_file(rpc_path).__dict__
            #d_out["rpc"] = rpc_rpcm_to_geotiff_format(rpc.__dict__)

            # additional fields for depth supervision
            ba_kps_pts3d_path = os.path.join(output_dir, "ba_files/ba_params/pts3d.npy")
            shutil.copyfile(ba_kps_pts3d_path, os.path.join(output_dir, "pts3d.npy"))
            cam_idx = ba_geotiff_basenames.index(d["img"])
            d["keypoints"] = {"2d_coordinates": ba_kps_pts2d[ba_kps_cam_ind == cam_idx, :].tolist(),
                              "pts3d_indices": ba_kps_pts3d_ind[ba_kps_cam_ind == cam_idx].tolist()}
        else:
            # use original rpc
            d["rpc"] = original_rpc.__dict__

        loader.save_dict_to_json(d, os.path.join(output_dir, "{}.json".format(loader.get_id(rgb_p))))

def create_train_test_splits(input_sample_ids, test_percent=0.15, min_test_samples=2):

    def shuffle_array(array):
        import random
        v = array.copy()
        random.shuffle(v)
        return v

    n_samples = len(input_sample_ids)
    input_sample_ids = np.array(input_sample_ids)
    all_indices = shuffle_array(np.arange(n_samples))
    n_test = max(min_test_samples, int(test_percent * n_samples))
    n_train = n_samples - n_test

    train_indices = all_indices[:n_train]
    test_indices = all_indices[-n_test:]

    train_samples = input_sample_ids[train_indices].tolist()
    test_samples = input_sample_ids[test_indices].tolist()

    return train_samples, test_samples

def create_satellite_dataset(aoi_id, dfc_dir, output_dir, ba=True):

    if ba:
        run_ba(aoi_id, dfc_dir, output_dir)
    create_dataset_from_DFC2019_data(aoi_id, dfc_dir, output_dir, use_ba=ba)

    # create train and test splits
    json_files = [os.path.basename(p) for p in glob.glob(os.path.join(output_dir, "*.json"))]
    train_samples, test_samples = create_train_test_splits(json_files)
    with open(os.path.join(output_dir, "train.txt"), "w+") as f:
        f.write("\n".join(train_samples))
    with open(os.path.join(output_dir, "test.txt"), "w+") as f:
        f.write("\n".join(test_samples))

if __name__ == '__main__':
    import fire
    fire.Fire(create_satellite_dataset)