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

To create the conda environments you can use the setup scripts, e.g.
```
conda init && bash -i setup_satnerf_env.sh
```

2. It is recommended to install `dsmr`. Otherwise the code will not crash but DSM registration will lose accuracy and affect the estimated altitude MAE.

Warning: If some libraries are not found, it may be necessary to update the environment variable `LD_LIBRARY_PATH` before launching the code:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```
where `$CONDA_PREFIX` is the path to your conda or miniconda environment (e.g. `/mnt/cdisk/roger/miniconda3/envs/satnerf`).

You can download [here](https://drive.google.com/drive/folders/1l0Jx0-MrvmDd8WBpGcxDo8_KWLpKfGhK?usp=sharing) the training and test datasets, as well as some pretrained models.

---

## 2. Testing

Example command:
```shell
(satnerf) $ python3 eval_satnerf.py 2022-03-24_22-50-35_JAX_068_ds1_sat-nerf /mnt/cdisk/roger/nerf_output-crops3/logs /mnt/cdisk/roger/nerf_output-crops3/results 24 val
```
---

## 3. Training

Example command:
```shell
(satnerf) $ python3 main.py --model sat-nerf --exp_name JAX_068_ds1_sat-nerf --root_dir /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth --logs_dir /mnt/cdisk/roger/Datasets/SatNeRF_output/logs --ckpts_dir /mnt/cdisk/roger/Datasets/SatNeRF_output/ckpts
```
---



## 4. Other functionalities


### 4.1. Dataset creation from the DFC2019 data:

The `create_satellite_dataset.py` script can be used to generate input datasets for SatNeRF from the open-source [DFC2019 data](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019). The `Track3-RGB` and `Track3-Truth` folders are needed.

We encourage you to use the `bundle_adjust` package, available [here](https://github.com/centreborelli/sat-bundleadjust), to ensure your dataset employs highly accurate RPC camera models. This will also allow aggregating depth supervision to the training and consequently boost the performance of the NeRF model.
```shell
(ba) $ python3 create_satellite_dataset.py JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/Datasets/SatNeRF_/root_dir/crops_rpcs_ba/JAX_068
```

Alternatively, if you prefer not installing `bundle_adjust`, it is also possible to use the flag `--noba` to create the dataset using the original RPC camera models from the DFC2019 data.
```shell
(ba) $ python3 create_satellite_dataset.py JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/Datasets/SatNeRF_/root_dir/crops_rpcs_raw/JAX_068 --noba
```

### 4.2. Depth supervision:

The script `check_depth_supervision_points.py` produces an interpolated DSM with the initial depths given by the 3D keypoints output by `bundle_adjust`.

Example command:
```shell
(satnerf) $ python3 study_depth_supervision.py 2022-03-25_12-59-21_JAX_068_ds1_sat-nerf_SCx0.1 /mnt/cdisk/roger/nerf_output-crops3/logs /mnt/cdisk/roger/nerf_output-crops3/results
```


### 4.3. Interpolate over different sun directions:

The script `eval_sun_interp.py` can be used to visualize images of the same AOI rendered with different solar direction vectors.

Example command:
```shell
(satnerf) $ python3 study_solar_correction.py 2022-03-25_12-59-21_JAX_068_ds1_sat-nerf_SCx0.1 /mnt/cdisk/roger/nerf_output-crops3/logs /mnt/cdisk/roger/nerf_output-crops3/results 24
```


### 4.4. Comparison to classic satellite MVS
We compare the DSMs learned by SatNeRF with the equivalent DSMs obtained from manually selected multiple stereo pairs, reconstructed using the [S2P](https://github.com/centreborelli/s2p) pipeline.
More details of the classic satellite MVS reconstruction process can be found [here](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Facciolo_Automatic_3D_Reconstruction_CVPR_2017_paper.html).
Use the script `eval_s2p.py` to reconstruct an AOI using this methodology.
```shell
(s2p) $ python3 eval_s2p.py JAX_068 /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/fullaoi_rpcs_ba_v1/JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/nerf_output-crops3/results --n_pairs 10
```