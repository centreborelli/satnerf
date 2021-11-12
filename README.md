# Sat-Nerf

##
### Training:

Example:
```
python3 main.py --exp_name depthx2_newpoints --root_dir /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068_ba_depth --img_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-RGB/JAX_068 --config_name s-nerf --cache_dir /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068_ba_depth/cache_512 --img_downscale 4 --checkpoints_dir /mnt/cdisk/roger/nerf_output/checkpoints --logs_dir /mnt/cdisk/roger/nerf_output/logs --gt_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth --depth_supervision
```

##
### Testing:

Usage: python3 eval_aoi.py run_id logs_dir output_dir epoch_number [checkpoints_dir]

Example:
```
python3 eval_aoi.py JAX_068 2021-11-03_08-00-59_depthx2_newpoints /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results 5
```

##
### Depth supervision:

It is possible to incorporate depth supervision to the training by using the following flags:

* `--depth_supervision`: Use a set of known 3d points to supervise the geometry learning
* `--depthloss_drop`: Epoch at which the depth supervision loss should be dropped [default: 10]
* `--depthloss_without_weights`: Do not use reprojection errors to weight depth supervision loss

The script check_depth_supervision_points.py can be used as a sanity check. It produces an interpolated DSM with the initial depths given by the 3D keypoints produced by bundle adjustment.

Usage: python3 eval_aoi.py run_id logs_dir output_dir

Example:
```
python3 check_depth_supervision_points.py 2021-11-03_08-00-59_depthx2_newpoints /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results
```

##
### Plot the evolution of the 3D reconstruction error:

Example:
```
python3 plot_depth_mae.py run_ids.txt /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth/JAX_068_DSM.tif 50 mae_evolution.png
```

##
### Create satellite dataset from the DFC2019 data:

Using bundle adjustment (allows aggregating depth supervision to the NeRF):
```
python3 create_satellite_dataset.py JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068_ba
```

It is also possible to use `--noba` to create the dataset using the original RPC camera models:
```
python3 create_satellite_dataset.py JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068 --noba
```
