# Sat-Nerf

Developed by the [Centre Borelli](https://www.centreborelli.fr/), ENS Paris-Saclay (2021).

---

## 1. Installation
1. Create a conda environment. GDAL is needed to handle DSM rasters efficiently.
```
conda create -n sat-nerf -c conda-forge python=3.6 libgdal
```
2. Activate the conda environment:
```
conda activate sat-nerf
```
3. Install the required Python packages. If you get an error similar to `Cannot uninstall X. It is a distutils installed project`, try adding the flag `--ignore-installed`.
```
 pip install -r requirements.txt
```
4. Install Pytorch as explained [here](https://github.com/pytorch/pytorch/issues/49161). The code was developed to run on a TITAN V or a RTX 3090.
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
5. It is recommended to install `dsmr`. Otherwise the code will not crash but DSM registration will lose accuracy and affect the estimated altitude MAE.

Extra 1: If some libraries are not found, it may be necessary to update the environment variable `LD_LIBRARY_PATH` before launching the code. Example:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/cdisk/roger/miniconda3/envs/sat-nerf/lib
```
Extra 2: To deactivate the conda environment use `conda deactivate`.  To remove the conda environment:
`conda env remove -n sat-nerf`.

---

## 2. Training:

Example:
```
python3 main.py --exp_name depthx2_newpoints --root_dir /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068_ba_depth --img_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-RGB/JAX_068 --config_name s-nerf --cache_dir /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068_ba_depth/cache_512 --img_downscale 4 --checkpoints_dir /mnt/cdisk/roger/nerf_output/checkpoints --logs_dir /mnt/cdisk/roger/nerf_output/logs --gt_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth --depth_supervision
```
---

## 3. Testing:

Usage: python3 eval_aoi.py run_id logs_dir output_dir epoch_number [checkpoints_dir]

Example:
```
python3 eval_aoi.py JAX_068 2021-11-03_08-00-59_depthx2_newpoints /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results 5
```
---

## 4. Other scripts


### 4.1. Dataset creation from the DFC2019 data:

The `create_satellite_dataset.py` script can be used to generate input datasets for Sat-NeRF from the open-source [DFC2019 data](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019). The `Track3-RGB` and `Track3-Truth` folders are needed.

We encourage you to use the `bundle_adjust` package, available [here](https://github.com/centreborelli/sat-bundleadjust), to ensure your dataset employs highly accurate RPC camera models. This will also allow aggregating depth supervision to the training and consequently boost the performance of the NeRF model. Example:
```
python3 create_satellite_dataset.py JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068_ba
```

Alternatively, if you prefer not installing `bundle_adjust`, it is also possible to use the flag `--noba` to create the dataset using the original RPC camera models from the DFC2019 data. Example:
```
python3 create_satellite_dataset.py JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068 --noba
```

### 4.2. Depth supervision:

If you used `bundle_adjust` to create your dataset (Section 4.1), it is possible to incorporate depth supervision to the training command (Section 2) by using the following flags:

* `--depth_supervision`: Use a set of known 3d points to supervise the geometry learning
* `--depthloss_drop`: Epoch at which the depth supervision loss should be dropped [default: 10]
* `--depthloss_without_weights`: Do not use reprojection errors to weight depth supervision loss

The script `check_depth_supervision_points.py` can additionally be used as a sanity check. It produces an interpolated DSM with the initial depths given by the 3D keypoints output by `bundle_adjust`.

Usage: python3 eval_aoi.py run_id logs_dir output_dir

Example:
```
python3 check_depth_supervision_points.py 2021-11-03_08-00-59_depthx2_newpoints /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results
```


### 4.3. Plot the evolution of the 3D reconstruction error:

The `plot_depth_mae.py` script may be useful to visualize how the altitude mean absolute error evolved during the training of one or multiple experiments.

Example:
```
python3 plot_depth_mae.py run_ids.txt /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth/JAX_068_DSM.tif 50 mae_evolution.png
```


### 4.4. Compare to s2p

Example:
```
python3 eval_s2p.py JAX_068 /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068 /mnt/cdisk/roger/Datasets/DFC2019
```

---