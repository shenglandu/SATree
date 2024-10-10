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
#include <iostream>
#include <fstream>


TreeSeg::TreeSeg()
        : other_points_(new Cloud3D)
        , tree_points_(new Cloud3D)
        , db_min_pts_(50)
        , db_radius_(0.2)
        , other_id_(0)
        , stem_id_(1)
        , crown_id_(2)
        , ignore_id_(-1)
        , eps_s_(0.5)
        , cut_height_(2.0)
        , radius_(0.15)
{
}


TreeSeg::~TreeSeg() {
    if (other_points_){
        other_points_->clear();
        other_points_ = nullptr;
    }
    if (tree_points_){
        tree_points_->clear();
        tree_points_ = nullptr;
        tree_props_.clear();
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
        int sem_id = pred[i];

        // Push the point into corresponding clouds according to its predicted semantic code
        if (sem_id == other_id_ or sem_id == ignore_id_){
            other_points_->push_back(pi);
        }
        else{
            tree_points_->push_back(pi);
            std::array<float, 5> prop{{float(sem_id), scores[i], dx[i], dy[i], dz[i]}};
            tree_props_.push_back(prop);
        }
    }

    std::cout << "Scene point statistics:" << std::endl;
    std::cout << "Total point number: " << x.size() << std::endl;
    std::cout << "Background point: " << other_points_->size() << std::endl;
    std::cout << "Tree point: " << tree_points_->size() << std::endl;

    return true;
}


bool TreeSeg::extract_stems() {
    // Initialize tree root containers
    roots_.clear();

    // Retrieve 2d stem points from the tree point cloud with the indices
    Cloud3D::Ptr proj_stem_pts(new Cloud3D);
    std::vector<int> stem_idx, noise_idx;
    for (int i = 0; i < tree_points_->size(); i++){
        int sem_id = int(tree_props_[i][0]);
        if (sem_id == stem_id_){
            Point3D pi = (*tree_points_)[i];
            pi.z = 0;
            proj_stem_pts->push_back(pi);
            stem_idx.push_back(i);
        }
    }

    // Perform clustering over stem points
    std::vector<Indices> proj_stem_clusters;
    OctreeEuclideanClusterExtraction<Point3D> oec;
    oec.setClusterTolerance(db_radius_);
    oec.setMinClusterSize(db_min_pts_);
    oec.setInputCloud(proj_stem_pts);
    oec.extract(proj_stem_clusters);

    // Construct projected tree point cloud and its octree
    Cloud3D::Ptr proj_tree_pts(new Cloud3D);
    copyPointCloud(*tree_points_, *proj_tree_pts);
    for (int i = 0; i < proj_tree_pts->size(); i++)
        (*proj_tree_pts)[i].z = 0;
    unibn::Octree<Point3D> octree;
    octree.initialize(*proj_tree_pts);
    std::vector<std::vector<uint32_t>> neighbors(2);

    // Check the stem candidate cluster quality based on score, height, and local maxima condition
    std::vector<int> stem_ind_container;
    for (int i = 0; i < proj_stem_clusters.size(); ++i) {
        // initialize the stem candidate cluster info
        int lowest_idx, highest_idx;
        float lowest = 1000;
        float highest = -1000;
        float avg_s, max_s1 = 0.;
        Point3D centroid(0, 0, 0);
        neighbors.clear();
        stem_ind_container.clear();

        // retrieve over per point in the stem cluster
        int count = 0;
        for (int& j: proj_stem_clusters[i].indices){
            int stem_ind = stem_idx[j];
            centroid.x += (*proj_stem_pts)[j].x;
            centroid.y += (*proj_stem_pts)[j].y;
            avg_s += tree_props_[stem_ind][1];
            float zj = (*tree_points_)[stem_ind].z;
            if (zj < lowest){
                lowest = zj;
                lowest_idx = stem_ind;
            }
            if (zj > highest){
                highest = zj;
                highest_idx = stem_ind;
            }
            if (tree_props_[stem_ind][1] > max_s1)
                max_s1 = tree_props_[stem_ind][1];
            stem_ind_container.push_back(stem_ind);
            count++;
        }
        avg_s = avg_s / float(count);
        centroid.x = centroid.x / float(count);
        centroid.y = centroid.y / float(count);

        // check if the score is sufficiently high
        if (avg_s <= eps_s_){
            // retrieve the 2-ring spherical neighborhoods from centroid to 2d tree points
            octree.radiusNeighbors<unibn::L2Distance<Point3D>>(centroid, radius_, neighbors[0]);
            octree.radiusNeighbors<unibn::L2Distance<Point3D>>(centroid, 3 * radius_, neighbors[1]);
            if (neighbors[0].empty() or neighbors[1].empty()){
                noise_idx.insert(noise_idx.end(), stem_ind_container.begin(), stem_ind_container.end());
                continue;
            }
            // collect scores of the 1st neighbourhood and obtain the max score value
            std::vector<float> neigh_s1, neigh_s2;
            for (auto& ni: neighbors[0]){
                if ((*tree_points_)[ni].z - highest <= 0.5)
                    neigh_s1.push_back(tree_props_[ni][1]);
            }
            float neigh_max_s1 = *std::max_element(neigh_s1.begin(), neigh_s1.end());
            max_s1 = std::max(neigh_max_s1, max_s1);
            // check if the 2nd neighbour contains crown points, collect scores in the 2nd neighbour
            int num_crown = 0;
            for (auto& ni: neighbors[1]){
                if ((*tree_points_)[ni].z - highest <= 0.5){
                    neigh_s2.push_back(tree_props_[ni][1]);
                    int sem_id = int(tree_props_[ni][0]);
                    if (sem_id == crown_id_)
                        num_crown ++;
                }
            }
            // if the neighbourhood contains no crown, then the stem is invalid
            if (num_crown == 0){
                noise_idx.insert(noise_idx.end(), stem_ind_container.begin(), stem_ind_container.end());
                continue;
            }
            // if any score in 2nd neighbor is greater than local maximal, then the stem is invalid
            float max_s2 = *std::max_element(neigh_s2.begin(), neigh_s2.end());
            float neigh_avg_s1 = std::accumulate(neigh_s1.begin(), neigh_s1.end(), 0.0) / neigh_s1.size();
            if (neigh_avg_s1 <= eps_s_){
                if (max_s1 <= 0.5 * eps_s_ or max_s2 > max_s1){
                    noise_idx.insert(noise_idx.end(), stem_ind_container.begin(), stem_ind_container.end());
                    continue;
                }
            }
        }

        // obtain the root location by stem points below the cutting height
        Point3D root(0, 0, lowest);
        count = 0;
        for (int& j: proj_stem_clusters[i].indices){
            int stem_ind = stem_idx[j];
            float zj = (*tree_points_)[stem_ind].z;
            if (zj - lowest <= cut_height_){
                root.x += (*proj_stem_pts)[j].x;
                root.y += (*proj_stem_pts)[j].y;
                count ++;
            }
        }
        root.x = root.x / float(count);
        root.y = root.y / float(count);

        // check if the root is too high above ground
        roots_.push_back(root);
    }

    ofstream mfile;
    mfile.open("/mnt/materials/PROJECT#3_Tree_Segmentation/Code/0_Preprocessing/Tree_Clouds/root/treeml/campus_root.xyz");
    for (int i = 0; i < roots_.size(); i++)
        mfile << roots_[i].x << " " << roots_[i].y << " " << roots_[i].z << endl;
    mfile.close();

    std::cout << roots_.size() << " number of roots have been extracted" << std::endl;
    proj_stem_pts->clear();
    proj_tree_pts->clear();

    return true;
}