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

#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include "tree_seg.h"


int main(int argc, char** argv) {

    std::string scene_file_nm = "/mnt/materials/PROJECT#3_Tree_Segmentation/Code/0_Preprocessing/Tree_Clouds/treeml/2023-01-09_tum_campus_0.ply";
//    std::string scene_file_nm(argv[1]);
//    std::string scene_output_nm(argv[2]);

    std::cout << "================================================" << std::endl;
    std::cout << "1. LOADING SCENE POINT CLOUDS FROM STRUCTURENET:" << std::endl;
    // Make sure the file is ply format
    size_t find = scene_file_nm.find(".ply");
    if (find == -1){
        std::cout << ".ply format required!" << std::endl;
        exit(EXIT_FAILURE);
    }

    TreeSeg *treeSeg = new TreeSeg();
    if (!treeSeg->read_clouds(scene_file_nm)){
        std::cout << "fail to read the clouds" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "================================================" << std::endl;
    std::cout << "2. EXTRACT TREE STEMS:" << std::endl;
    if (!treeSeg->extract_stems()){
        std::cout << "fail to extract stems" << std::endl;
        exit(EXIT_FAILURE);
    }



    return EXIT_SUCCESS;
}

