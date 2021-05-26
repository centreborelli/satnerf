"""
This script defines the input parameters that can be customized from the command line
"""

import argparse
import datetime

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="directory to save model checkpoints.")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="directory to save experiment logs.")
    parser.add_argument("--config_name", type=str, default="s-nerf_basic",
                        choices=['nerf', 's-nerf_basic'],
                        help="NeRF training and model configuration")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Pretrained checkpoint path to load")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")

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

    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum iteration steps for training')

    args = parser.parse_args()

    exp_id = args.config_name if args.exp_name is None else args.exp_name
    args.exp_name = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), exp_id)


    return args