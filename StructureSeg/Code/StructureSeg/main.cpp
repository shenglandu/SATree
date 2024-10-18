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

#include <iostream>
#include "tree_seg.h"


int main(int argc, char** argv) {

    std::string config_nm = "/mnt/materials/PROJECT#3_Tree_Segmentation/Code/StructureSeg/Config/treeml.ini";
//    std::string scene_file_nm(argv[1]);

    // Initialize a tree seg object
    TreeSeg *treeSeg = new TreeSeg();
    treeSeg->initialize(config_nm);

    std::cout << "================================================" << std::endl;
    std::cout << "1. LOADING SCENE POINT CLOUDS FROM STRUCTURENET:" << std::endl;
    if (!treeSeg->read_clouds()){
        std::cout << "fail to read the clouds" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "================================================" << std::endl;
    std::cout << "2. EXTRACT TREE STEMS:" << std::endl;
    if (!treeSeg->extract_stems()){
        std::cout << "fail to extract stems" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "================================================" << std::endl;
    std::cout << "3. GROUP TREE INSTANCES:" << std::endl;
    if (!treeSeg->group_trees()){
        std::cout << "fail to group tree points" << std::endl;
        exit(EXIT_FAILURE);
    }
    treeSeg->output_tree_seg();

    return EXIT_SUCCESS;
}

