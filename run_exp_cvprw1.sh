# run experiments

aoi_id=$1
suffix=$2
gpu_id=1
downsample_factor=4
errs="$aoi_id"_errors.txt
DFC2019_dir="/mnt/cdisk/roger/Datasets/DFC2019"
satnerf_data_dir="/mnt/cdisk/roger/Datasets/nerf_satellite"
out_dir="/mnt/cdisk/roger/nerf_output"

echo -n "" > $errs 

if [ "$downsample_factor" = "4" ]; then
    imsize=512
elif [ "$downsample_factor" = "8" ]; then
    imsize=256
else
    imsize=1024 
fi

# (1) to justify the irradiance model

# classic NeRF
exp_name="$aoi_id"_"$imsize"_classic
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id"_ba_v2 --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name nerf --cache_dir $satnerf_data_dir/"$aoi_id"_ba_v2/cache_$imsize --img_downscale $downsample_factor --exp_name "$exp_name"_"$suffix" --checkpoints_dir $out_dir/ckpts_cvprw_2 --logs_dir $out_dir/logs_cvprw_2 --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 2>> $errs

# shadow-NeRF (without SC)
exp_name="$aoi_id"_"$imsize"_snerf
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id"_ba_v2 --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf --cache_dir $satnerf_data_dir/"$aoi_id"_ba_v2/cache_$imsize --img_downscale $downsample_factor --exp_name "$exp_name"_"$suffix" --checkpoints_dir $out_dir/ckpts_cvprw_2 --logs_dir $out_dir/logs_cvprw_2 --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 2>> $errs

# sat-NeRF (without beta)
exp_name="$aoi_id"_"$imsize"_satnerf
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id"_ba_v2 --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf-w --cache_dir $satnerf_data_dir/"$aoi_id"_ba_v2/cache_"$imsize" --img_downscale $downsample_factor --exp_name "$exp_name"_"$suffix" --checkpoints_dir $out_dir/ckpts_cvprw_2 --logs_dir $out_dir/logs_cvprw_2 --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 2>> $errs


# (2) to justify the complete cost function

# sat-NeRF + uncertainty
exp_name="$aoi_id"_"$imsize"_satnerf_beta
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id"_ba_v2 --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf-w --cache_dir $satnerf_data_dir/"$aoi_id"_ba_v2/cache_$imsize --img_downscale $downsample_factor --exp_name "$exp_name"_"$suffix" --checkpoints_dir $out_dir/ckpts_cvprw_2 --logs_dir $out_dir/logs_cvprw_2 --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 --uncertainty 2>> $errs

# shadow-NeRF + solar correction
exp_name="$aoi_id"_"$imsize"_satnerf_beta_SCx0.05
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id"_ba_v2 --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf --cache_dir $satnerf_data_dir/"$aoi_id"_ba_v2/cache_$imsize --img_downscale $downsample_factor --exp_name "$exp_name"_"$suffix" --checkpoints_dir $out_dir/ckpts_cvprw_2 --logs_dir $out_dir/logs_cvprw_2 --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0.05 2>> $errs


# (3) to justify the benefits of bundle adjustment 

# sat-NeRF without bundle adjustment
exp_name="$aoi_id"_"$imsize"_satnerf_beta_noBA
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id" --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf-w --cache_dir $satnerf_data_dir/"$aoi_id"/cache_"$imsize" --img_downscale $downsample_factor --exp_name "$exp_name"_"$suffix" --checkpoints_dir $out_dir/ckpts_cvprw_2 --logs_dir $out_dir/logs_cvprw_2 --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 --uncertainty 2>> $errs

# sat-NeRF with bundle adjustment and depth supervision
exp_name="$aoi_id"_"$imsize"_satnerf_beta_DS
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id"_ba_v2 --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf-w --cache_dir $satnerf_data_dir/"$aoi_id"_ba_v2/cache_$imsize --img_downscale $downsample_factor --exp_name "$exp_name"_"$suffix" --checkpoints_dir $out_dir/ckpts_cvprw_2 --logs_dir $out_dir/logs_cvprw_2 --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 --depth_supervision --depthloss_lambda 1000 --uncertainty #2>> $errs


# sat nerf regularization
#python3 main.py --root_dir $satnerf_data_dir/$aoi_id --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf-w --cache_dir $satnerf_data_dir/$aoi_id/cache_"$imsize"_reg --img_downscale $downsample_factor --exp_name "$exp_name"_"$suffix" --checkpoints_dir $out_dir/checkpoints --logs_dir $out_dir/logs --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 --patches

