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

#include "tree_seg.h"
#include <pcl/common/angles.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "voxel_grid_fix.h"
#include "octree_extract_clusters.h"
#include "CSF.h"
#include "octree_unibn.hpp"
#include <vtkOBBTree.h>


TreeSeg::TreeSeg()
        : other_points_(new Cloud3D)
        , crown_points_(new Cloud3D)
        , stem_points_(new Cloud3D)
        , db_min_pts_(50)
        , db_radius_(0.2)
        , other_id_(0)
        , stem_id_(1)
        , crown_id_(2)
{
}


TreeSeg::~TreeSeg() {
    if (other_points_){
        other_points_->clear();
        other_points_ = nullptr;
    }
    if (crown_points_){
        crown_points_->clear();
        crown_points_ = nullptr;
        crown_props_.clear();
    }
    if (stem_points_){
        stem_points_->clear();
        stem_points_ = nullptr;
        stem_props_.clear();
    }
}


bool TreeSeg::read_clouds(const std::string &file_nm) {
    // Read the ply file
    PLYData plyIn(file_nm);

    // Access the data, coordinates, scores, offsets, and GT instance label
    std::vector<float> x = plyIn.getElement("vertex").getProperty<float>("x");
    std::vector<float> y = plyIn.getElement("vertex").getProperty<float>("y");
    std::vector<float> z = plyIn.getElement("vertex").getProperty<float>("z");
    std::vector<float> scores = plyIn.getElement("vertex").getProperty<float>("s");
    std::vector<float> dx = plyIn.getElement("vertex").getProperty<float>("dx");
    std::vector<float> dy = plyIn.getElement("vertex").getProperty<float>("dy");
    std::vector<float> dz = plyIn.getElement("vertex").getProperty<float>("dz");
    std::vector<int> pred = plyIn.getElement("vertex").getProperty<int>("preds");
    // std::vector<int> ins = plyIn.getElement("vertex").getProperty<int>("ins");

    // Read cloud point by point
    for (int i = 0; i < x.size(); i++){
        // Read point coordinates, attributes (score and offsets), and semantic prediction
        Point3D pi(x[i], y[i], z[i]);
        std::array<float, 4> prop{ {scores[i], dx[i], dy[i], dz[i]}};
        int sem_id = pred[i];

        // Push the point into corresponding clouds according to its predicted semantic code
        if (sem_id == other_id_){
            other_points_->push_back(pi);
        }
        else if (sem_id == stem_id_){
            stem_points_->push_back(pi);
            stem_props_.push_back(prop);
        }
        else if (sem_id == crown_id_){
            crown_points_->push_back(pi);
            crown_props_.push_back(prop);
        }
        else
            continue;
    }

    std::cout << "Scene point statistics:" << std::endl;
    std::cout << "Total point number: " << x.size() << std::endl;
    std::cout << "Background point: " << other_points_->size() << std::endl;
    std::cout << "Tree stem point: " << stem_points_->size() << std::endl;
    std::cout << "Tree crown point: " << crown_points_->size() << std::endl;

}