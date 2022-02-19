import os
import numpy as np
import glob
import rasterio
import rpcm
import json

def RGB2YUV(rgb):
    # input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
    # output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
    # source: https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
    m = np.array([[0.29900, -0.16874, 0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[:, :, 1:] += 128.0
    return yuv

def YUV2RGB(yuv):
    # input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
    # output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
    # source: https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304
    return rgb

def norm_rgb(x, mi=None, ma=None):
    if mi is None:
        mi = np.nanmin(x)
    if ma is None:
        ma = np.nanmax(x)
    x = (x-mi)/(ma-mi) # normalize to 0~1
    x = np.clip(x, 0, 1)
    #x = (255.*x).astype(np.uint8)
    x = (255. * x)
    x = np.clip(x, 0, 255)
    return x

def preprocess_rgb(aoi_id, dfc_dir):
    # the rgb images provided in the DFC2019 dataset are compressed to [0-255] and contain saturated pixels
    # we can recover the texture in saturated areas by replacing the luminance channel with the panchromatic image

    if aoi_id in ["JAX_004", "JAX_068"]:
        pan_dir = "/vsicurl/http://138.231.80.166:2332/grss-2019/track_3/Track3-MSI-1/"
    else:
        pan_dir = "/vsicurl/http://138.231.80.166:2332/grss-2019/track_3/Track3-MSI-3/"

    all_img = True

    out_rgb_dir = os.path.join(dfc_dir, "Track3-RGB_HR/{}".format(aoi_id))
    os.makedirs(out_rgb_dir, exist_ok=True)
    in_rgb_dir = os.path.join(dfc_dir, "Track3-RGB/{}".format(aoi_id))
    if all_img:
        myimages = sorted(glob.glob(in_rgb_dir + "/*.tif"))
    else:
        myimages = [sorted(glob.glob(in_rgb_dir + "/*.tif"))[0]]
    n_img = len(myimages)
    new_min, new_max = np.inf, -np.inf
    d = {}
    for img_idx, in_rgb_path in enumerate(myimages):
        pan_path = os.path.join(pan_dir, os.path.basename(in_rgb_path).replace("RGB.tif", "PAN.tif"))
        with rasterio.open(in_rgb_path, 'r') as f:
            rgb_in = np.transpose(f.read(), (1, 2, 0))
            profile = f.profile
            tags = f.tags()
        rpc = rpcm.rpc_from_geotiff(in_rgb_path)
        with rasterio.open(pan_path, 'r') as f:
            pan = np.transpose(f.read(), (1, 2, 0))[:, :, 0]
            p_max, p_min = np.percentile(pan, 99.5), np.percentile(pan, 0.0)
            pan = np.clip(pan, p_min, p_max)
            pan = pan ** (1/1.5) #np.sqrt(pan)

        yuv = RGB2YUV(rgb_in)
        y = yuv[:, :, 0]
        med = np.nanmedian((pan / y).flatten())
        d[os.path.basename(in_rgb_path)] = med
        pan_ = pan / med
        shadows = abs(pan_ - y) > 20
        pan_[shadows] = y[shadows]
        
        print("avg diff: {:.2f}".format(np.mean(abs(pan / med - y).ravel())))

        yuv[:, :, 0] = pan_
        rgb_out = YUV2RGB(yuv)
        new_min = min(np.nanmin(rgb_out.ravel()), new_min)
        new_max = max(np.nanmax(rgb_out.ravel()), new_max)

        # write output rgb
        out_rgb_path = os.path.join(out_rgb_dir, os.path.basename(in_rgb_path))
        profile["dtype"] = rgb_out.dtype
        with rasterio.open(out_rgb_path, 'w', **profile) as dst:
            dst.write(np.transpose(rgb_out, (2, 0, 1)))
            dst.update_tags(**tags)
            dst.update_tags(ns='RPC', **rpc.to_geotiff_dict())
        print("{}/{}".format(img_idx + 1, n_img))
    d["min_value"] = new_min
    d["max_value"] = new_max


    # re-write all images in the range 0-255
    print("re-writing everything in range [0-255]...")
    myimages = sorted(glob.glob(out_rgb_dir + "/*.tif"))
    for p in myimages:
        with rasterio.open(p, 'r') as f:
            rgb = np.transpose(f.read(), (1, 2, 0))
            profile = f.profile
            tags = f.tags()
        rpc = rpcm.rpc_from_geotiff(p)
        os.remove(p)
        rgb = norm_rgb(rgb, mi=0, ma=300)
        with rasterio.open(p, 'w', **profile) as dst:
            dst.write(np.transpose(rgb, (2, 0, 1)))
            dst.update_tags(**tags)
            dst.update_tags(ns='RPC', **rpc.to_geotiff_dict())
    print("done!")


    with open(os.path.join(out_rgb_dir, "debug.json"), "w") as f:
        json.dump(d, f, indent=2)

    print("creating crops...")
    if all_img:
        myimages = sorted(glob.glob(out_rgb_dir + "/*.tif"))
    else:
        myimages = [sorted(glob.glob(out_rgb_dir + "/*.tif"))[0]]
    from create_satellite_dataset import crop_geotiff_lonlat_aoi, read_DFC2019_lonlat_aoi
    aoi_lonlat = read_DFC2019_lonlat_aoi(aoi_id, dfc_dir)
    crops_dir = os.path.join(dfc_dir, "Track3-RGB-crops_HR4/{}".format(aoi_id))
    os.makedirs(crops_dir, exist_ok=True)
    for p in myimages:
        crop_geotiff_lonlat_aoi(p, os.path.join(crops_dir, os.path.basename(p)), aoi_lonlat)
    print("done!")	
if __name__ == '__main__':
    import fire
    fire.Fire(preprocess_rgb)


