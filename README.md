# SparseBEV

This is the official PyTorch implementation for paper [SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos](https://arxiv.org/abs/2308.09244). (ICCV 2023)

## Model Zoo

| Backbone | Pretrain | Input Size | Epochs | Training Cost | NDS | FPS | Config | Weights |
|----------|----------|------------|--------|---------------|-----|-----|--------|---------|
| R50 | [nuImages](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/nuimages/cascade-mask-rcnn_r50_fpn_coco-20e_20e_nuim.py) | 704x256 | 36 | 28h (8x2080Ti) | 55.8 | 23.5 | [config](configs/r50_nuimg_704x256_400q_36ep.py) | [weights](https://drive.google.com/file/d/1C_Vn3iiSnSW1Dw1r0DkjJMwvHC5Y3zTN/view?usp=sharing) |

* FPS is measured on a machine with AMD 5800X and RTX 3090.
* The noise is around 0.3 NDS.

## Environment

Install PyTorch 2.0 + CUDA 11.8:

```
conda create -n sparsebev python=3.8
conda activate sparsebev
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

or PyTorch 1.10.2 + CUDA 10.2 for older GPUs:

```
conda create -n sparsebev python=3.8
conda activate sparsebev
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=10.2 -c pytorch
```

Install other dependencies:

```
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
```

Install turbojpeg and pillow-simd to speed up data loading (optional but important):

```
sudo apt-get update
sudo apt-get install -y libturbojpeg
pip install pyturbojpeg
pip uninstall pillow
pip install pillow-simd==9.0.0.post1
```

Compile CUDA extensions:

```
cd models/csrc
python setup.py build_ext --inplace
```

## Prepare Dataset

1. Download nuScenes from [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes) and put it in `data/nuscenes`.
2. Download the generated info file from [Google Drive](https://drive.google.com/file/d/1uyoUuSRIVScrm_CUpge6V_UzwDT61ODO/view?usp=sharing) and unzip it.
3. Folder structure:

```
data/nuscenes
├── maps
├── nuscenes_infos_test_sweep.pkl
├── nuscenes_infos_train_mini_sweep.pkl
├── nuscenes_infos_train_sweep.pkl
├── nuscenes_infos_val_mini_sweep.pkl
├── nuscenes_infos_val_sweep.pkl
├── samples
├── sweeps
├── v1.0-mini
├── v1.0-test
└── v1.0-trainval
```

These `*.pkl` files can also be generated with our script: `gen_sweep_info.py`.

## Training

Train SparseBEV with 8 GPUs:

```
torchrun --nproc_per_node 8 train.py --config configs/r50_nuimg_704x256_400q_36ep.py
```

Train SparseBEV with 4 GPUs (i.e the last four GPUs):

```
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node 4 train.py --config configs/r50_nuimg_704x256_400q_36ep.py
```

The batch size for each GPU will be scaled automatically. So there is no need to modify the `batch_size` in config files.

## Evaluation

Single-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0
python val.py --config configs/r50_nuimg_704x256_400q_36ep.py --weights checkpoints/r50_nuimg_704x256_400q_36ep.pth
```

Multi-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 val.py --config configs/r50_nuimg_704x256_400q_36ep.py --weights checkpoints/r50_nuimg_704x256_400q_36ep.pth
```

## Timing

FPS is measured with a single GPU:

```
export CUDA_VISIBLE_DEVICES=0
python timing.py --config configs/r50_nuimg_704x256_400q_36ep.py --weights checkpoints/r50_nuimg_704x256_400q_36ep.pth
```

## Acknowledgements

Many thanks to these excellent open-source projects:

* 3D Detection: [DETR3D](https://github.com/WangYueFt/detr3d), [PETR](https://github.com/megvii-research/PETR), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [StreamPETR](https://github.com/exiawsh/StreamPETR)
* 2D Detection: [AdaMixer](https://github.com/MCG-NJU/AdaMixer), [DN-DETR](https://github.com/IDEA-Research/DN-DETR)
* Codebase: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow)