#!/usr/bin/env bash

PROJECT_DIR=~/jupyter/satnerf
EXP_DIR=~/jupyter/satnerf/exp
TRAIN_STEPS=6250
BATCH_SIZE=32768
CHUNK=65536
EXP_NAME=JAX_068_ds1_nerf_debug

if [ ! -d "$EXP_DIR/$EXP_NAME" ]; then
  mkdir "$EXP_DIR/$EXP_NAME"
fi

python3 main.py --root_dir $PROJECT_DIR/datasets/root_dir/crops_rpcs_raw/JAX_068 \
                --img_dir $PROJECT_DIR/datasets/DFC2019/Track3-RGB-crops/JAX_068 \
                --gt_dir $PROJECT_DIR/datasets/DFC2019/Track3-Truth \
                --exp_name $EXP_NAME \
                --model nerf \
                --img_downscale 1 \
                --cache_dir $EXP_DIR/$EXP_NAME/cache/crops_rpcs_raw/JAX_068_ds1 \
                --logs_dir $EXP_DIR/$EXP_NAME/logs \
                --ckpts_dir $EXP_DIR/$EXP_NAME/checkpoints \
                --gpu_id 2 \
                --max_train_steps $TRAIN_STEPS \
                --batch_size $BATCH_SIZE \
                --chunk $CHUNK \
                --fc_units 256