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
#include "happly.h"


using namespace happly;

class TreeSeg
{
public:
        TreeSeg();
        ~TreeSeg();

        //----------------------------------- Variables -------------------------------------
public:
        // Name and path of the scene
        std::string scene_name_;
        std::string scene_path_;

        // Clustering parameters
        int db_min_pts_;
        float db_radius_;

        // Semantic code
        int other_id_;
        int stem_id_;
        int crown_id_;
        int ignore_id_;

        // Stem selection parameters
        float eps_s_;
        float cut_height_;
        float radius_;

        // Voxelization size
        float grid_size_;

        // Tree grouping parameters
        float scale_;
        float eps_dist_;
        float shrink_ratio_;

        // Bool indicator of outputting root positions
        bool is_output_root_;

private:
        // Scene background points
        Cloud3D::Ptr other_points_;

        // Scene tree points and properties
        Cloud3D::Ptr tree_points_;
        std::vector<std::array<float, 5>> tree_props_;  // semantic prediction, score, offset

        // Scene noise indices
        std::vector<int> noise_idx_;

        // Tree point to voxel map indices
        std::vector<int> tree_voxel_map_idx_;
        std::vector<std::vector<int>> voxel_idx_;

        // Tree root positions and tree root cluster indices
        std::vector<Point3D> roots_;
        std::vector<std::vector<int>> roots_idx_;
        std::vector<int> tree_root_idx_;
        std::vector<GraphVertexDescriptor> root_vertices_;

        // Tree pseudo root (for shortest path searching, removed afterwards)
        Point3D pseudo_root_;
        GraphVertexDescriptor pseudo_root_vertex_;

        // Tree ID colormap
        std::vector<std::array<int, 3>> colormap_;

        // Tree graphs
        Graph delaunay_;
        Graph MST_;

        //------------------------------------ Methods --------------------------------------
public:
        // Initialize the configuration
        void initialize(const std::string &config_nm);

        // Parse file name to get scene name and scene path
        bool parse_scene_name(const std::string &file_nm);

        // Read the cloud points from the file
        bool read_clouds();

        // Extract the tree stems
        bool extract_stems();

        // Group tree points
        bool group_trees();

        // Output tree segmentation
        void output_tree_seg();

        // Output tree root positions
        void output_root_xyz();

private:
        // Voxelize tree points
        void voxelize_tree_points();

        // Filter extracted roots based on height
        void filter_roots();

        // Build Delaunay graph
        void build_delaunay();

        // Extract MST graph
        void extract_mst();

        // Assign tree id to graph vertices
        void assign_tree_id();

        // Compute graph weights
        void compute_graph_weights();

        // Obtain root vertex
        void obtain_root_vertex();

        // Shift the coordinates
        Point3D shift_point_3d(Point3D p, Point3D dir, double s);

        // Normalize a vector
        Point3D normalize_point_3d(Point3D p);

        // Calculate the distance between points
        double compute_pair_distance(Point3D p1, Point3D p2);

        // Calculate the distance between points
        double compute_pair_distance_2d(Point3D p1, Point3D p2);

};


#endif
