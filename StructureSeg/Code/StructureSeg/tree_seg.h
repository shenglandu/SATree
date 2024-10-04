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
        // Name of the scene
        std::string scene_name_;

        // Clustering parameters
        int db_min_pts_;
        float db_radius_;

        // Semantic code
        int other_id_;
        int stem_id_;
        int crown_id_;

private:
        // Scene background cloud
        Cloud3D::Ptr other_points_;

        // Scene crown cloud
        Cloud3D::Ptr crown_points_;
        std::vector<std::array<float, 4>> crown_props_;  // score, offset

        // Scene stem cloud
        Cloud3D::Ptr stem_points_;
        std::vector<std::array<float, 4>> stem_props_;  // score, offset

        // Tree root positions
        std::vector<Point3D> roots_;
        std::vector<GraphVertexDescriptor> root_vertices_;

        // Tree pseudo root (for shortest path searching, removed afterwards)
        Point3D pseudo_root_;
        GraphVertexIterator pseudo_root_vertex_;

        // Tree graphs
        Graph delaunay_;
        Graph MST_;

        //------------------------------------ Methods --------------------------------------
public:
        // Read the cloud points from the file
        bool read_clouds(const std::string &file_nm);

        // Extract the tree stems
        bool extract_stems();

};


#endif
