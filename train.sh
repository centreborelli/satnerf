#!/usr/bin/env bash

export project_dir=~/jupyter/satnerf
export exp_dir=~/jupyter/satnerf/exp
export exp_name=JAX_068_ds1_nerf_2gpu_batch2048

if [ ! -d "$exp_dir/$exp_name" ]; then
  mkdir "$exp_dir/$exp_name"
fi

python3 main.py --root_dir $project_dir/datasets/root_dir/crops_rpcs_raw/JAX_068 \
                --img_dir $project_dir/datasets/DFC2019/Track3-RGB-crops/JAX_068 \
                --gt_dir $project_dir/datasets/DFC2019/Track3-Truth \
                --exp_name $exp_name \
                --model nerf \
                --img_downscale 1 \
                --cache_dir $exp_dir/$exp_name/cache/crops_rpcs_raw/JAX_068_ds1 \
                --logs_dir $exp_dir/$exp_name/logs \
                --ckpts_dir $exp_dir/$exp_name/checkpoints \
                --gpu_id 2 \
                --max_train_steps 50000 \
                --batch_size 2048 \
                --chunk 4096 \
                --fc_units 256 2>> $exp_dir/$exp_name/outputs.txt