"""
This script defines the input parameters that can be customized from the command line
"""

import argparse
import datetime
import json
import os

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="directory to save model checkpoints.")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="directory to save experiment logs.")
    parser.add_argument("--config_name", type=str, default="s-nerf",
                        choices=['nerf', 's-nerf'],
                        help="NeRF training and model configuration")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Pretrained checkpoint path to load")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--gpu_id", type=int, required=True,
                        help="GPU that will be used")

    # dataset options
    parser.add_argument('--dataset_name', type=str, default='satellite',
                        choices=['satellite', 'blender'],
                        help='Dataset type to train/val')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Directory where the images are located (if different than root_dir)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory where cache for the current dataset is found')
    parser.add_argument('--img_downscale', type=float, default=1.0,
                        help='Downscale factor for the input images')
    parser.add_argument('--depth_supervision', dest='depth', action='store_true',
                        help='Use a set of known 3d points to supervise the geometry learning')
    parser.add_argument('--depthloss_drop', type=float, default=0.1,
                        help='Portion of training steps at which the depth supervision loss should be dropped')
    parser.add_argument('--depthloss_without_weights', action='store_true',
                        help='Do not use reprojection errors to weight depth supervision loss')
    parser.add_argument('--depthloss_lambda', type=float, default=1.0,
                        help='Float that multiplies the sum of depth supervision errors')
    parser.add_argument('--solarloss_lambda', type=float, default=0.05,
                        help='Float that multiplies the solar correction loss terms')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Directory where the ground truth DSM is located (if available)')

    args = parser.parse_args()

    exp_id = args.config_name if args.exp_name is None else args.exp_name
    args.exp_name = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), exp_id)
    print("\nRunning {} - Using gpu {}\n".format(args.exp_name, args.gpu_id))

    os.makedirs("{}/{}".format(args.logs_dir, args.exp_name), exist_ok=True)
    with open("{}/{}/opts.json".format(args.logs_dir, args.exp_name), "w") as f:
        json.dump(vars(args), f, indent=2)

    return args
