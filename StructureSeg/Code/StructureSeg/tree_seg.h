/*
 * Software License Agreement (Apache License)
 *
 *  Copyright (C) 2024, Shenglan Du (shenglan.du@tudelft.nl),
 *                      Liangliang Nan (liangliang.nan@gmail.com).
 *  All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

#ifndef STRUCTURESEG_TREE_SEG_H
#define STRUCTURESEG_TREE_SEG_H

#include "common.h"


class TreeSeg
{
public:
        TreeSeg();
        ~TreeSeg();

private:

    Cloud3D::ConstPtr cloud_input_;
    Indices::Ptr indices_understory_;
    Cloud3D::Ptr cloud_stems_;

    // Subsampling
    float leaf_size_;
    // Verticality-based filtering
    float search_radius_;
    float verticality_threshold_;
    // Euclidean clustering
    int min_pts_per_cluster_;
    float min_dist_between_stems_;
};


#endif
