import torch
import yaml
import os
import json
import utils
from models import NeRF
from datasets import SatelliteDataset
from rendering import render_rays
from collections import defaultdict
from config import DefaultConfig, TrainingConfig, SNerfBasicConfig, SNerfWBasicConfig
import metrics
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)

@torch.no_grad()
def batched_inference(models, rays, ts, conf):
    """Do batched inference on rays using chunk."""
    chunk_size = conf.training.chunk
    batch_size = rays.shape[0]

    results = defaultdict(list)
    for i in range(0, batch_size, chunk_size):
        rendered_ray_chunks = \
            render_rays(models, conf, rays[i:i + chunk_size],
                        ts[i:i + chunk_size] if ts is not None else None)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        if results[k][0] is None:
            results[k] = None
        else:
            results[k] = torch.cat(v, 0)

    return results

def load_nerf(run_id, log_path, checkpoints_dir, epoch_number, args):

    checkpoint_path = os.path.join(checkpoints_dir, "{}/epoch={}.ckpt".format(run_id, epoch_number))
    print("Using", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Could not find checkpoint {}".format(checkpoint_path))

    with open("{}/version_0/hparams.yaml".format(log_path), 'r') as stream:
        conf_dict = yaml.safe_load(stream)
        if conf_dict["name"] == "nerf":
            conf = DefaultConfig(**conf_dict)
        if conf_dict["name"] == "s-nerf":
            conf = SNerfBasicConfig(**conf_dict)
        if conf_dict["name"] == "s-nerf-w":
            conf = SNerfWBasicConfig(**conf_dict)
        conf.training = TrainingConfig(**conf.training)

    # load models
    models = {}

    if conf.name == "s-nerf-w":
        embedding_t = torch.nn.Embedding(conf.N_vocab, conf.N_tau)
        load_ckpt(embedding_t, checkpoint_path, model_name='embedding_t')
        models["t"] = embedding_t.cuda('cuda:0').cuda()
        t_embbeding_size = conf.N_tau
    else:
        t_embbeding_size = 0

    nerf_coarse = NeRF(layers=conf.layers,
                       feat=conf.feat,
                       input_sizes=conf.input_sizes,
                       skips=conf.skips,
                       siren=conf.siren,
                       mapping=conf.mapping,
                       mapping_sizes=conf.mapping_sizes,
                       variant=conf.name,
                       t_embedding_dims=t_embbeding_size,
                       predict_uncertainty=args["uncertainty"])
    load_ckpt(nerf_coarse, checkpoint_path, model_name='nerf_coarse')
    nerf_coarse.cuda('cuda:0').eval()
    models["coarse"] = nerf_coarse
    if conf.n_importance > 0:
        nerf_fine = NeRF(layers=conf.layers,
                         feat=conf.feat,
                         input_sizes=conf.input_sizes,
                         skips=conf.skips,
                         siren=conf.siren,
                         mapping=conf.mapping,
                         mapping_sizes=conf.mapping_sizes,
                         variant=conf.name,
                         t_embedding_dims=t_embbeding_size,
                         predict_uncertainty=args["uncertainty"])
        load_ckpt(nerf_fine, checkpoint_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models["fine"] = nerf_fine
    return models, conf

def save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number):

    rays = sample["rays"].squeeze()
    rgbs = sample["rgbs"].squeeze()
    src_path = sample["src_path"][0]
    src_id = sample["src_id"][0]

    typ = "fine" if "rgb_fine" in results else "coarse"
    if "h" in sample and "w" in sample:
        W, H = sample["w"][0], sample["h"][0]
    else:
        W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))  # assume squared images
    print(results[f'rgb_{typ}'].shape, W, H)
    img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    depth = results[f"depth_{typ}"]

    # save depth prediction
    _, _, alts = dataset.get_latlonalt_from_nerf_prediction(rays.cpu(), depth.cpu())
    out_path = "{}/depth/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    utils.save_output_image(alts.reshape(1, H, W), out_path, src_path)
    # save dsm
    out_path = "{}/dsm/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    dsm = dataset.get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
    # save rgb image
    out_path = "{}/rgb/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    utils.save_output_image(img, out_path, src_path)
    # save gt rgb image
    out_path = "{}/gt_rgb/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    utils.save_output_image(img_gt, out_path, src_path)
    # save shadow modelling images
    if f"sun_{typ}" in results:
        s_v = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'sun_{typ}'], -2)
        out_path = "{}/sun/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        utils.save_output_image(s_v.view(1, H, W).cpu(), out_path, src_path)
        rgb_albedo = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'albedo_{typ}'], -2)
        out_path = "{}/albedo/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        utils.save_output_image(rgb_albedo.cpu().view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)
        if f"ambient_a_{typ}" in results:
            a_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'ambient_a_{typ}'], -2)
            out_path = "{}/ambient_a/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            utils.save_output_image(a_rgb.view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)
            b_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'ambient_b_{typ}'], -2)
            out_path = "{}/ambient_b/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            utils.save_output_image(b_rgb.view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)
        if f"beta_{typ}" in results:
            beta = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'beta_{typ}'], -2)
            out_path = "{}/beta/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            utils.save_output_image(beta.view(1, H, W).cpu(), out_path, src_path)
        if f"sky_{typ}" in results:
            sky_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'sky_{typ}'], -2)
            out_path = "{}/sky/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            utils.save_output_image(sky_rgb.cpu().view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)

def compute_mae_and_save_dsm_errs(pred_dsm_path, roi_path, gt_dsm_path, out_dir, src_id, epoch_number):
    # save dsm errs
    from metrics import dsm_pointwise_abs_errors
    if os.path.exists(roi_path) and os.path.exists(gt_dsm_path):
        roi_metadata = np.loadtxt(roi_path)
        out_path1 = "{}/dsm_errs/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        out_path2 = "{}/dsm_errs2/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        abs_err = dsm_pointwise_abs_errors(pred_dsm_path, gt_dsm_path, roi_metadata, gt_mask_path=None,
                                           out_rdsm_path=out_path2, out_err_path=out_path1)
    return np.nanmean(abs_err.ravel())

def find_best_embbeding_for_val_image(models, rays, conf, gt_rgbs, train_indices=None):

    best_ts = None
    best_psnr = 0.

    if train_indices is None:
        train_indices = torch.arange(conf.N_vocab)
    for t in train_indices:
        ts = t.long() * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        results = batched_inference(models, rays, ts, conf)
        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), gt_rgbs.cpu())
        if psnr_ > best_psnr:
            best_ts = ts
            best_psnr = psnr_

    return best_ts

def find_best_embeddings_for_val_dataset(val_dataset, models, conf, train_indices):
    print("finding best embedding indices for validation dataset...")
    list_of_image_indices = [0]
    for i in np.arange(1, len(val_dataset)):
        sample = val_dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_id = sample["src_id"]
        aoi_id = src_id[:7]
        if aoi_id in ["JAX_068", "JAX_004", "JAX_214"]:
            t = predefined_val_ts(src_id)
        else:
            ts = find_best_embbeding_for_val_image(models, rays, conf, rgbs, train_indices=train_indices)
            t = torch.unique(ts).cpu().numpy()
        print("{}: {}".format(src_id, t))
        list_of_image_indices.append(t)
    print("... done!")
    return list_of_image_indices

def predefined_val_ts(img_id):

    aoi_id = img_id[:7]

    if aoi_id == "JAX_068":
        d = {"JAX_068_013_RGB": 0,
             "JAX_068_002_RGB": 8,
             "JAX_068_012_RGB": 1} #3
    elif aoi_id == "JAX_004":
        d = {"JAX_004_022_RGB": 0,
             "JAX_004_014_RGB": 0,
             "JAX_004_009_RGB": 5}
    elif aoi_id == "JAX_214":
        d = {"JAX_214_020_RGB": 0,
             "JAX_214_006_RGB": 8,
             "JAX_214_001_RGB": 18,
             "JAX_214_008_RGB": 2}
    elif aoi_id == "JAX_156":
        d = {"JAX_156_009_RGB": 0,
             "JAX_156_003_RGB": 8,
             "JAX_156_004_RGB": 14,
             "JAX_156_014_RGB": 18}
    elif aoi_id == "JAX_260":
        d = {"JAX_260_015_RGB": 0,
             "JAX_260_006_RGB": 3,
             "JAX_260_004_RGB": 10}
    else:
        return None
    return d[img_id]



def eval_aoi(run_id, logs_dir, output_dir, epoch_number, split, checkpoints_dir=None):

    #gpu_idx = 1
    #run_id = "2021-07-14_00-00-00_snerfw_mask_attempt2"
    #epoch_number = "15"

    log_path = os.path.join(logs_dir, run_id)
    with open('{}/opts.json'.format(log_path), 'r') as f:
            args = json.load(f)

    # load nerf
    if checkpoints_dir is None:
        checkpoints_dir = args["checkpoints_dir"]
    models, conf = load_nerf(run_id, log_path, checkpoints_dir, epoch_number-1, args)

    # load dataset
    dataset = SatelliteDataset(args["root_dir"], args["img_dir"], split="val",
                               img_downscale=args["img_downscale"], cache_dir=args["cache_dir"])

    with open(os.path.join(args["root_dir"], "train.txt"), "r") as f:
        train_jsons = f.read().split("\n")
        n_train = len(train_jsons)

    if conf.name == "s-nerf-w":
        train_indices = torch.arange(n_train)
        if split == "val":
            val_t_indices = find_best_embeddings_for_val_dataset(dataset, models, conf, train_indices)
        else:
            val_t_indices = train_indices.numpy()
    else:
        if split == "val":
            val_t_indices = [None]*len(dataset)
        else:
            val_t_indices = [None]*n_train

    if split == "train":
        with open(os.path.join(args["root_dir"], "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        dataset.json_files = [os.path.join(args["root_dir"], json_p) for json_p in json_files]
        dataset.all_ids = [i for i, p, in enumerate(dataset.json_files)]
        samples_to_eval = np.arange(0, len(dataset))
    else:
        samples_to_eval = np.arange(1, len(dataset))

    psnr = []
    ssim = []
    mae = []

    for i in samples_to_eval:
        sample = dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_id = sample["src_id"]

        if val_t_indices[i] is None:
            ts = None
        else:
            ts = val_t_indices[i] * torch.ones(rays.shape[0], 1).long().cuda().squeeze()

        results = batched_inference(models, rays, ts, conf)

        for k in sample.keys():
            if torch.is_tensor(sample[k]):
                sample[k] = sample[k].unsqueeze(0)
            else:
                sample[k] = [sample[k]]
        out_dir = os.path.join(output_dir, "eval_aoi", run_id, split)
        os.makedirs(out_dir, exist_ok=True)
        save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number)

        # image metrics
        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), rgbs.cpu())
        psnr.append(psnr_)
        W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))
        ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W).cpu(), rgbs.view(1, 3, H, W).cpu())
        ssim.append(ssim_)

        # geometry metrics
        pred_dsm_path = "{}/dsm/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        aoi_id = src_id[:7]
        roi_path = os.path.join(args["gt_dir"], aoi_id + "_DSM.txt")
        gt_dsm_path = os.path.join(args["gt_dir"], aoi_id + "_DSM.tif")
        mae_ = compute_mae_and_save_dsm_errs(pred_dsm_path, roi_path, gt_dsm_path, out_dir, src_id, epoch_number)
        mae.append(mae_)
        os.system(f"rm tmp*.tif.xml")

        print("{}: pnsr {:.3f} / ssim {:.3f} / mae {:.3f}".format(src_id, psnr_, ssim_, mae_))

    print("\nMean PSNR: {:.3f}".format(np.mean(np.array(psnr))))
    print("Mean SSIM: {:.3f}".format(np.mean(np.array(ssim))))
    print("Mean MAE: {:.3f}\n".format(np.mean(np.array(mae))))

if __name__ == '__main__':
    import fire
    fire.Fire(eval_aoi)
