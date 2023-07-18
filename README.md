# Sat-NeRF
---
**UPDATE JUNE 2023 !!!** Have a look at [EO-NeRF](https://rogermm14.github.io/eonerf/), our latest method for multi-view satellite photogrammetry using neural radiance fields.

---

### [[Project page]](https://centreborelli.github.io/satnerf)

Developed at the [ENS Paris-Saclay, Centre Borelli](https://www.centreborelli.fr/) and accepted at the [CVPR EarthVision Workshop 2022](https://www.grss-ieee.org/events/earthvision-2022/).

### [Sat-NeRF: Learning Multi-View Satellite Photogrammetry With Transient Objects and Shadow Modeling Using RPC Cameras](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Mari_Sat-NeRF_Learning_Multi-View_Satellite_Photogrammetry_With_Transient_Objects_and_Shadow_CVPRW_2022_paper.pdf)
*[Roger Marí](https://scholar.google.com/citations?user=TgpSmIsAAAAJ&hl=en), 
[Gabriele Facciolo](http://dev.ipol.im/~facciolo/),
[Thibaud Ehret](https://tehret.github.io/)*

> **Abstract:** *We introduce the Satellite Neural Radiance Field (Sat-NeRF), a new end-to-end model for learning multi-view satellite photogrammetry in the wild. Sat-NeRF combines some of the latest trends in neural rendering with native satellite camera models, represented by rational polynomial coefficient (RPC) functions. The proposed method renders new views and infers surface models of similar quality to those obtained with traditional state-of-the-art stereo pipelines. Multi-date images exhibit significant changes in appearance, mainly due to varying shadows and transient objects (cars, vegetation). Robustness to these challenges is achieved by a shadow-aware irradiance model and uncertainty weighting to deal with transient phenomena that cannot be explained by the position of the sun. We evaluate Sat-NeRF using WorldView-3 images from different locations and stress the advantages of applying a bundle adjustment to the satellite camera models prior to training. This boosts the network performance and can optionally be used to extract additional cues for depth supervision.*

If you find this code or work helpful, please cite:
```
@inproceedings{mari2022sat,
  title={{Sat-NeRF}: Learning Multi-View Satellite Photogrammetry With Transient Objects and Shadow Modeling Using {RPC} Cameras},
  author={Mar{\'\i}, Roger and Facciolo, Gabriele and Ehret, Thibaud},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={1310-1320},
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

Warning: If some libraries are not found, it may be necessary to update the environment variable `LD_LIBRARY_PATH` before launching the code:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```
where `$CONDA_PREFIX` is the path to your conda or miniconda environment (e.g. `/mnt/cdisk/roger/miniconda3/envs/satnerf`).

You can download [here](https://github.com/centreborelli/satnerf/releases/tag/EarthVision2022) the training and test datasets, as well as some pretrained models.

---

## 2. Testing

Example command to generate a surface model with Sat-NeRF:
```shell
(satnerf) $ export dataset_dir=/mnt/cdisk/roger/EV2022_satnerf/dataset
(satnerf) $ export pretrained_models=/mnt/cdisk/roger/EV2022_satnerf/pretrained_models
(satnerf) $ python3 create_satnerf_dsm.py Sat-NeRF $pretrained_models/JAX_068 out_dsm_path/JAX_068 28 $pretrained_models/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 $dataset_dir/DFC2019/Track3-Truth
```

Example command for novel view synthesis with Sat-NeRF:
```shell
(satnerf) $ python3 eval_satnerf.py Sat-NeRF $pretrained_models/JAX_068 out_eval_path/JAX_068 28 val $pretrained_models/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 $dataset_dir/DFC2019/Track3-Truth
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
(ba) $ python3 create_satellite_dataset.py JAX_068 $dataset_dir/DFC2019 out_dataset_path/JAX_068
```

Alternatively, if you prefer not installing `bundle_adjust`, it is also possible to use the flag `--noba` to create the dataset using the original RPC camera models from the DFC2019 data.
```shell
(ba) $ python3 create_satellite_dataset.py JAX_068 $dataset_dir/DFC2019 out_dataset_path/JAX_068 --noba
```
The `--splits` flag can also be used to generate the `train.txt` and `test.txt` files.

### 4.2. Depth supervision:

The script `study_depth_supervision.py` produces an interpolated DSM with the initial depths given by the 3D keypoints output by `bundle_adjust`.

Example command:
```shell
(satnerf) $ python3 study_depth_supervision.py Sat-NeRF+DS $pretrained_models/JAX_068 out_DS_study_path/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops $dataset_dir/DFC2019/Track3-Truth
```


### 4.3. Interpolate over different sun directions:

The script `study_solar_interpolation.py` can be used to visualize images of the same AOI rendered with different solar direction vectors.

Example command:
```shell
(satnerf) $ python3 study_solar_interpolation.py Sat-NeRF $pretrained_models/JAX_068 out_solar_study_path/JAX_068 28 $pretrained_models/JAX_068 $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 $dataset_dir/DFC2019/Track3-Truth
```


### 4.4. Comparison to classic satellite MVS
We compare the DSMs learned by SatNeRF with the equivalent DSMs obtained from manually selected multiple stereo pairs, reconstructed using the [S2P](https://github.com/centreborelli/s2p) pipeline.
More details of the classic satellite MVS reconstruction process can be found [here](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w18/html/Facciolo_Automatic_3D_Reconstruction_CVPR_2017_paper.html).
Use the script `eval_s2p.py` to reconstruct an AOI using this methodology.
```shell
(s2p) $ python3 eval_s2p.py JAX_068 /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/fullaoi_rpcs_ba_v1/JAX_068 /mnt/cdisk/roger/Datasets/DFC2019 /mnt/cdisk/roger/nerf_output-crops3/results --n_pairs 10
```
