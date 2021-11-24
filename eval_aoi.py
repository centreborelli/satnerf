import torch
import yaml
import os
import json
import utils
from models import NeRF
from datasets import SatelliteDataset
from rendering import render_rays
from collections import defaultdict
from config import SNerfBasicConfig, TrainingConfig
import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
def batched_inference(models, rays, conf):
    """Do batched inference on rays using chunk."""
    chunk_size = conf.training.chunk
    batch_size = rays.shape[0]

    results = defaultdict(list)
    for i in range(0, batch_size, chunk_size):
        rendered_ray_chunks = \
            render_rays(models,
                        rays[i:i + chunk_size],
                        conf=conf,
                        chunk=chunk_size,
                        test_time=False)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    return results


def eval_aoi(run_id, logs_dir, output_dir, epoch_number, checkpoints_dir=None):

    #gpu_idx = 1
    #run_id = "2021-07-14_00-00-00_snerfw_mask_attempt2"
    #epoch_number = "15"

    log_path = os.path.join(logs_dir, run_id)
    with open('{}/opts.json'.format(log_path), 'r') as f:
            args = json.load(f)

    if checkpoints_dir is None:
        checkpoints_dir = args["checkpoints_dir"]
    checkpoint_path = os.path.join(checkpoints_dir, "{}/epoch={}.ckpt".format(run_id, epoch_number))
    print("Using", checkpoint_path)

    with open("{}/version_0/hparams.yaml".format(log_path), 'r') as stream:
        conf_dict = yaml.safe_load(stream)
        if conf_dict["name"] =="s-nerf":
            conf = SNerfBasicConfig(**conf_dict)
        conf.training = TrainingConfig(**conf.training)

    # load dataset
    dataset = SatelliteDataset(args["root_dir"], args["img_dir"], split="val",
                               img_downscale=args["img_downscale"], cache_dir=args["cache_dir"])

    # load models
    models = {}
    nerf_coarse = NeRF(layers=conf.layers,
                       feat=conf.feat,
                       input_sizes=conf.input_sizes,
                       skips=conf.skips,
                       siren=conf.siren,
                       mapping=conf.mapping,
                       mapping_sizes=conf.mapping_sizes,
                       variant=conf.name)
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
                         variant=conf.name)
        load_ckpt(nerf_fine, checkpoint_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models["fine"] = nerf_fine

    out_dir = os.path.join(output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(dataset)):
        sample = dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_path = sample["src_path"]
        src_id = sample["src_id"]

        results = batched_inference(models, rays, conf)

        typ = "fine" if "rgb_fine" in results else "coarse"
        W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))
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
            s_v = torch.sum(results[f"weights_{typ}"] * results[f"sun_{typ}"], -1)
            out_path = "{}/sun/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            utils.save_output_image(s_v.cpu().reshape(1, H, W), out_path, src_path)
            rgb_sky = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'sky_{typ}'], -2)
            out_path = "{}/sky/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            utils.save_output_image(rgb_sky.view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)
            rgb_albedo = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'albedo_{typ}'], -2)
            out_path = "{}/albedo/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            utils.save_output_image(rgb_albedo.cpu().view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)

        psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), rgbs.cpu())
        print("{}: pnsr {:.2f}".format(src_id, psnr_))

if __name__ == '__main__':
    import fire
    fire.Fire(eval_aoi)