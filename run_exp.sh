# run experiments

aoi_id=$1
gpu_id=0
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

# Classic NeRF
exp_name="$aoi_id"_"$imsize"_classic
python3 main.py --root_dir $satnerf_data_dir/$aoi_id --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name nerf --cache_dir $satnerf_data_dir/$aoi_id/cache_$imsize --img_downscale $downsample_factor --exp_name $exp_name --checkpoints_dir $out_dir/checkpoints --logs_dir $out_dir/logs --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 2>> $errs

# Shadow-NeRF
exp_name="$aoi_id"_"$imsize"_snerf
python3 main.py --root_dir $satnerf_data_dir/$aoi_id --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf --cache_dir $satnerf_data_dir/$aoi_id/cache_$imsize --img_downscale $downsample_factor --exp_name $exp_name --checkpoints_dir $out_dir/checkpoints --logs_dir $out_dir/logs --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 2>> $errs

# Sat-NeRF
exp_name="$aoi_id"_"$imsize"_satnerf
python3 main.py --root_dir $satnerf_data_dir/$aoi_id --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf-w --cache_dir $satnerf_data_dir/$aoi_id/cache_$imsize --img_downscale $downsample_factor --exp_name $exp_name --checkpoints_dir $out_dir/checkpoints --logs_dir $out_dir/logs --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 2>> $errs

# Sat-NeRF + BA
exp_name="$aoi_id"_"$imsize"_satnerf_BA
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id"_ba --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf-w --cache_dir $satnerf_data_dir/"$aoi_id"_ba/cache_$imsize --img_downscale $downsample_factor --exp_name $exp_name --checkpoints_dir $out_dir/checkpoints --logs_dir $out_dir/logs --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 2>> $errs

# Sat-NeRF + BA + DS
exp_name="$aoi_id"_"$imsize"_satnerf_BA_DS
python3 main.py --root_dir $satnerf_data_dir/"$aoi_id"_ba --img_dir $DFC2019_dir/Track3-RGB/$aoi_id --config_name s-nerf-w --cache_dir $satnerf_data_dir/"$aoi_id"_ba/cache_$imsize --img_downscale $downsample_factor --exp_name $exp_name --checkpoints_dir $out_dir/checkpoints --logs_dir $out_dir/logs --gt_dir $DFC2019_dir/Track3-Truth --gpu_id $gpu_id --solarloss_lambda 0 --depth_supervision 2>> $errs
