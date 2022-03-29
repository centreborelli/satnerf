# run experiments

aoi_id=$1
suffix=$2
gpu_id=1
downsample_factor=1
training_iters=500000
errs="$aoi_id"_errors.txt
DFC2019_dir="/mnt/cdisk/roger/Datasets/DFC2019"
root_dir="/mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/$aoi_id"
cache_dir="/mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/"$aoi_id"_ds"$downsample_factor
img_dir=$DFC2019_dir/Track3-RGB-crops/$aoi_id
out_dir="/mnt/cdisk/roger/nerf_output-crops3"
logs_dir=$out_dir/logs
ckpts_dir=$out_dir/ckpts
errs_dir=$out_dir/errs
mkdir $errs_dir
gt_dir=$DFC2019_dir/Track3-Truth


# basic NeRF
model="nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --fc_units 256"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

# shadow NeRF
model="s-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

# shadow NeRF + solar correction
model="s-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"_SCx0.05
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --sc_lambda 0.05"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

#########################################################################

# satellite NeRF
model="sat-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args #2>> $errs

# satellite NeRF + solar correction
model="sat-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"_SCx0.1
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --sc_lambda 0.1"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

#########################################################################

# satellite NeRF + solar correction (without BA)
root_dir="/mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_raw/$aoi_id"
cache_dir="/mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_raw/"$aoi_id"_ds"$downsample_factor
model="sat-nerf"
exp_name=o_"$aoi_id"_ds"$downsample_factor"_"$model"_SCx0.1_noBA
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --sc_lambda 0.1"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs

# satellite NeRF + depth supervision
root_dir="/mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/$aoi_id"
cache_dir="/mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/"$aoi_id"_ds"$downsample_factor
model="sat-nerf"
exp_name=o_"$aoi_id"_ds"$downsample_factor"_"$model"_DSx1000
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --ds_lambda 1000"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 2>> $errs


