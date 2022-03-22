# SatNeRF

Developed by the [Centre Borelli](https://www.centreborelli.fr/), ENS Paris-Saclay (2021).

### [Sat-NeRF: Learning Multi-View Satellite Photogrammetry With Transient Objects and Shadow Modeling Using RPC Cameras](https://arxiv.org/abs/2203.08896)
*[Roger Marí](https://scholar.google.com/citations?user=TgpSmIsAAAAJ&hl=en), 
[Gabriele Facciolo](http://dev.ipol.im/~facciolo/),
[Thibaud Ehret](https://tehret.github.io/)*

> **Abstract:** *We introduce the Satellite Neural Radiance Field (Sat-NeRF), a new end-to-end model for learning multi-view satellite photogrammetry in the wild. Sat-NeRF combines some of the latest trends in neural rendering with native satellite camera models, represented by rational polynomial coefficient (RPC) functions. The proposed method renders new views and infers surface models of similar quality to those obtained with traditional state-of-the-art stereo pipelines. Multi-date images exhibit significant changes in appearance, mainly due to varying shadows and transient objects (cars, vegetation). Robustness to these challenges is achieved by a shadow-aware irradiance model and uncertainty weighting to deal with transient phenomena that cannot be explained by the position of the sun. We evaluate Sat-NeRF using WorldView-3 images from different locations and stress the advantages of applying a bundle adjustment to the satellite camera models prior to training. This boosts the network performance and can optionally be used to extract additional cues for depth supervision.*

If you find this code or work helpful, please cite:
```
@article{mari2022satnerf,
  title={{Sat-NeRF}: Learning Multi-View Satellite Photogrammetry With Transient Objects and Shadow Modeling Using {RPC} Cameras},
  author={Mar{\'\i}, Roger and Facciolo, Gabriele and Ehret, Thibaud},
  journal={arXiv preprint arXiv:2203.08896},
  year={2022}
}
```

---


## 1. Setup and Data
1. This project works with multiple conda environments, named `satnerf`, `s2p` and `ba`.

- `satnerf` is the only strictly necessary environment. It is required to train/test SatNeRF.
- `s2p` is used to additionally evaluate a satellite MVS pipeline relying on classic computer vision methods.
- `ba` is used to bundle adjust the RPCs of the DFC2019 data. 

To create the conda environments you can use:
```
conda init && bash -i create_conda_env.sh
```

2. It is recommended to install `dsmr`. Otherwise the code will not crash but DSM registration will lose accuracy and affect the estimated altitude MAE.

Warning: If some libraries are not found, it may be necessary to update the environment variable `LD_LIBRARY_PATH` before launching the code. E.g:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```
where `$CONDA_PREFIX` is the path to your conda or miniconda environment (e.g. `/mnt/cdisk/roger/miniconda3/envs/satnerf`).

You can download [here](https://drive.google.com/drive/folders/1l0Jx0-MrvmDd8WBpGcxDo8_KWLpKfGhK?usp=sharing) the training and test datasets, as well as some pretrained models.

The original open-source DFC2019 data can be downloaded [here](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019). Only the `Track3-RGB` and `Track3-Truth` folders are needed.


---

## 2. Testing

SatNeRF models can be tested using the following command.

Usage: python3 eval_aoi.py run_id logs_dir output_dir epoch_number [checkpoints_dir]

Example:
```shell
(satnerf) $ python3 eval_satnerf.py 2021-11-03_08-00-59_depthx2_newpoints /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results 5
```
---

## 3. Training

Example:
```shell
(satnerf) $ python3 main.py --model sat-nerf --exp_name exp_4 --root_dir /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth
```
---



## 4. Other functionalities


### 4.1. Dataset creation from the DFC2019 data:

The `create_satellite_dataset.py` script can be used to generate input datasets for SatNeRF from the open-source [DFC2019 data](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019). The `Track3-RGB` and `Track3-Truth` folders are needed.

We encourage you to use the `bundle_adjust` package, available [here](https://github.com/centreborelli/sat-bundleadjust), to ensure your dataset employs highly accurate RPC camera models. This will also allow aggregating depth supervision to the training and consequently boost the performance of the NeRF model. Example:
```shell
(ba) $ python3 create_satellite_dataset.py JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068_ba
```

Alternatively, if you prefer not installing `bundle_adjust`, it is also possible to use the flag `--noba` to create the dataset using the original RPC camera models from the DFC2019 data. Example:
```shell
(ba) $ python3 create_satellite_dataset.py JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068 --noba
```

### 4.2. Depth supervision:

If you used `bundle_adjust` to create your dataset (Section 4.1), it is possible to incorporate depth supervision to the training command (Section 2) by using the following flags:

* `--depth_supervision`: Use a set of known 3d points to supervise the geometry learning
* `--depthloss_drop`: Epoch at which the depth supervision loss should be dropped [default: 10]
* `--depthloss_without_weights`: Do not use reprojection errors to weight depth supervision loss

The script `check_depth_supervision_points.py` can additionally be used as a sanity check. It produces an interpolated DSM with the initial depths given by the 3D keypoints output by `bundle_adjust`.

Usage: python3 eval_aoi.py run_id logs_dir output_dir

Example:
```shell
(satnerf) $ python3 check_depth_supervision_points.py 2021-11-03_08-00-59_depthx2_newpoints /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results
```


### 4.3. Plot the evolution of the 3D reconstruction error:

The `plot_depth_mae.py` script may be useful to visualize how the altitude mean absolute error evolved during the training of one or multiple experiments.

Example:
```shell
(satnerf) $ python3 plot_depth_mae.py run_ids.txt /mnt/cdisk/roger/nerf_output/logs /mnt/cdisk/roger/nerf_output/results /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth/JAX_068_DSM.tif 50 mae_evolution.png
```


### 4.4. Comparison to classic satellite MVS
We compare the DSMs learned by SatNeRF with the equivalent DSMs obtained from manually selected multiple stereo pairs, reconstructed using the [S2P](https://github.com/centreborelli/s2p) pipeline.
More details of the classic satellite MVS reconstruction process can be found [here](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Facciolo_Automatic_3D_Reconstruction_CVPR_2017_paper.html).

To evaluate S2P please use the following command:
```shell
(s2p) $ python3 eval_s2p.py JAX_068 /mnt/cdisk/roger/Datasets/nerf_satellite/JAX_068 /mnt/cdisk/roger/Datasets/DFC2019
```