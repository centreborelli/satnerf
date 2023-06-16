#!/usr/bin/env bash

export project_dir=
python3 main.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir $project_dir/datasets/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir $project_dir/datasets/DFC2019/Track3-RGB-crops/JAX_068 \
                --gt_dir $project_dir/datasets/DFC2019/Track3-Truth \
                --cache_dir $project_dir/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --logs_dir project_dir/logs \
                --ckpts_dir project_dir/checkpoints \
                --gpu_id 0