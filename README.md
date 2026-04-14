# SATree: Structure-aware tree instance segmentation from 3D LiDAR point clouds

This repo is the implementation for [SATree: Structure-aware tree instance segmentation from 3D LiDAR point clouds](https://authors.elsevier.com/sd/article/S1618-8667(26)00154-8).

![overview](Fig/overview.png)

## Overview
We propose SATree, a novel structure-aware approach that directly identifies important tree structures, such as crowns and stems, from point clouds, enabling robust tree instance segmentation against tree overlaps and varying tree sizes. Our method leverages a multi-task learning framework that simultaneously performs (i) semantic segmentation to classify a point as crown, stem, or other; (ii) heatmap prediction to assign a heat value to each point based on 2D Gaussian kernels centered at tree stem locations; (iii) offset prediction to estimate point-wise offset vectors pointing to the instance centroid. Our research outputs are precisely segmented 3D tree instances that support downstream forestry inventory, 3D tree reconstruction, and fine-grained part segmentation of trees. 

## Data preprocessing instructions
### TreeML
For TreeML dataset, the original data can be downloaded from [this](https://springernature.figshare.com/collections/TreeML-Data_a_multidisciplinary_and_multilayer_urban_tree_dataset/6788358/1) link. We need three datasets:
- Dataset_tree
- Dataset_building_other
- Dataset_QSM

cd to `SATree/SANet/openpoints/dataset/treeml/prepare_treeml_strongstem.py`, specify the paths for these three datasets in L274-L276. Besides, you need to specify your own output ply path. Then, you can run preprocessing using:

        python prepare_treeml_strongstem.py

### ForInstance
For ForInstance dataset, the original data can be downloaded from [this](https://zenodo.org/records/8287792) link. Download the dataset without changing its structure. cd to `SATree/SANet/openpoints/dataset/forinstance/prepare_forinstance.py`, specify the tree path and output ply path in L171 and L175. Then, you can run preprocessing using:

        python prepare_forinstance.py

Note that the following packages are required to successfully run preprocessing scripts:
- pandas
- scikit-learn
- h5py
- laspy

## SANet training procedures
### Backbone reference
We adopt [PointMetaBase](https://arxiv.org/abs/2211.14462) as the backbone for point feature learning. Please refer to their [Pytorch implementation](https://github.com/linhaojia13/PointMetaBase) for installation and setup.

### Training on TreeML
cd to `SATree/SANet/cfgs/treeml/default.yaml`, specify the ply path of the previously processed data in the field `data_root`. Then, you can train SANet on TreeML using:

        CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/treeml/pointmetabase-l.yaml wandb.use_wandb=False

For testing, you can use:

        CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/treeml/pointmetabase-l.yaml wandb.use_wandb=False mode=test --pretrained_path [specify your pretrained weight here. By default, we use the ckpt_latest.pth for testing]

### Training on ForInstance
cd to `SATree/SANet/cfgs/forinstance/default.yaml`, specify the ply path of the previously processed data in the field `data_root`. Then, you can train SANet on TreeML using:

        CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/forinstance/pointmetabase-l.yaml wandb.use_wandb=False

For testing, you can use:

        CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/forinstance/pointmetabase-l.yaml wandb.use_wandb=False mode=test --pretrained_path [specify your pretrained weight here. By default, we use the ckpt_latest.pth for testing]

### Pretrained weights
We provide pretrained weights on both datasets, which can be accessed via the [drive](https://drive.google.com/drive/folders/18YBgdVIZ1jMNw1ZEuBPAJz-jGDFVM7CE?usp=sharing) link. Note that for the released weights on the ForInstance dataset, we have observed slightly better performance scores than the scores reported in the paper.

## SASeg instance segmentation steps
### Build
The implementation of SASeg depends on the packages of [PCL](https://pointclouds.org/), [VTK](https://vtk.org/), and [Boost](https://www.boost.org/). cd to `SATree/SASeg/Code/CMakeLists.txt`. In L27-28, you need to manually set the path to the Boost library. Then, you can build SASeg using compilers such as [CLion](https://www.jetbrains.com/clion/). Building requires CMake`>=3.12` and C++`>=14`.

### Run
Currently, the instance segmentation is performed on the single-scene level. For segmenting individual trees of a specific scene in the TreeML dataset, cd to `SATree/SASeg/Config/treeml.ini`, specify the path to the network prediction outputs (.ply) of the scene. Then, in `SATree/SASeg/Code/SASeg/main.cpp` L28, specify the path to this `treeml.ini` configuration file. You can run the instance segmentation of the given scene of the TreeML dataset. For the ForInstance dataset, you need to look at `SATree/SASeg/Config/forinstance.ini`.

Running SASeg outputs a "_seg.ply" file for an input scene, where it encodes the ground truth instance label, the predicted instance label, and the rgb for each scene point. If the field `is_output_root` is set as true, SASeg also outputs the extracted tree roots as an ".xyz" file.

## Evaluation
cd to `SATree/Tools/SAEval.py`. Specify the path to the folder where you have stored your instance segmentation results in L310. Then, you can specify the forest scene name in L311. Evaluation is performed using:

        python SAEval.py

## Citation
If you use the code and SATree's approach in a scientific work, please cite our paper:
```
@article{du2026satree,
  title={SATree: Structure-aware tree instance segmentation from 3D LiDAR point clouds},
  author={Du, Shenglan and Stoter, Jantien and Kooij, Julian F.P. and Nan, Liangliang},
  journal={Urban Forestry & Urban Greening},
  year={2026},
  volume={120},
  pages={129414},
  doi={10.1016/j.ufug.2026.129414}
}
```

## Acknowledgements
Our implementation of post-segmentation and quantitative evaluation is partially inspired by the following codebases:
- Yang et al., [GlobalMatch](https://github.com/zexinyang/GlobalMatch), 2023
- Jiang et al., [PointGroup](https://github.com/JIA-Lab-research/PointGroup), 2020


