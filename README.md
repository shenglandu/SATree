# SATree: Structure-aware tree instance segmentation from 3D LiDAR point clouds

This repo is the implementation for [SATree: Structure-aware tree instance segmentation from 3D LiDAR point clouds](https://authors.elsevier.com/sd/article/S1618-8667(26)00154-8).

![overview](img/overview.pdf)

## Abstract
Accurate segmentation and analysis of individual trees from 3D point clouds is a crucial yet challenging task in urbanism and environmental studies. Most existing methods for tree instance segmentation suffer from either under- or over-segmentation errors, mainly due to the complex nature of the environments and the varying tree geometries. In this paper, we propose SATree, a novel structure-aware approach that directly identifies important tree structures, such as crowns and stems, from point clouds, enabling robust tree instance segmentation against tree overlaps and varying tree sizes. Our method leverages a multi-task learning framework that simultaneously performs (i) semantic segmentation to classify a point as crown, stem, or other; (ii) heatmap prediction to assign a heat value to each point based on 2D Gaussian kernels centered at tree stem locations; (iii) offset prediction to estimate point-wise offset vectors pointing to the instance centroid. Key to our approach is the stem localization module, where we fuse the semantic and heatmap predictions to reliably localize tree stems from the network outputs. After that, we utilize a graph-based shortest path algorithm to group individual tree points by integrating the learned offset embeddings. Extensive experiments on two public forestry datasets, TreeML and ForInstance, demonstrate that SATree consistently outperforms state-of-the-art methods in terms of AP, AP50, and AP25 scores, reducing significant under- or over-segmentation errors. Our research output supports downstream forestry inventory, 3D tree reconstruction, and fine-grained part segmentation of trees. 

## Backbone Reference
We adopt [PointMetaBase](https://arxiv.org/abs/2211.14462) as the backbone for point feature learning. Please refer to their [Pytorch implementation](https://github.com/linhaojia13/PointMetaBase) for installation and setup.

## How to run SATree
Coming soon...

## Citation
If you use (part of) the code/approach in a scientific work, please cite our paper:
```
@article{du2026satree,
  title={SATree: Structure-aware tree instance segmentation from 3D LiDAR point clouds},
  author={Du, Shenglan and Stoter, Jantien and Kooij, Julian F.P. and Nan, Liangliang},
  journal={Urban Forestry &amp; Urban Greening},
  year={2026},
  doi={10.1016/j.ufug.2026.129414}
}
```

