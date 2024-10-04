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

    std::string scene_file_nm = "/mnt/materials/PROJECT#3_Tree_Segmentation/Code/0_Preprocessing/Tree_Clouds/new/2023-01-09_tum_campus_0.ply";
//    std::string scene_file_nm(argv[1]);
//    std::string scene_output_nm(argv[2]);

    //============================= Load Data =============================
    std::cout << "1. LOADING SCENE POINT CLOUDS FROM STRUCTURENET:" << std::endl;
    // Make sure the file is ply format
    size_t find = scene_file_nm.find(".ply");
    if (find == -1){
        std::cout << ".ply format required!" << std::endl;
        exit(EXIT_FAILURE);
    }

    TreeSeg *treeSeg = new TreeSeg();
    treeSeg->read_clouds(scene_file_nm);

//
//    //======================= Pairwise Registration =======================
//    std::cout << "2. MAPPING STEMS:" << std::endl;
//    Cloud3D::Ptr cloud_pos_src(new Cloud3D), cloud_pos_tgt(new Cloud3D);
//    // tic = omp_get_wtime();
//    Mapping mapping;
//    mapping.setInputCloud(cloud_src->makeShared());
//    mapping.extract(cloud_pos_src);
//    // toc = omp_get_wtime();
//    // time_val += (toc - tic);
//    print_info("  (1) ");
//    print_value("%d", cloud_pos_src->size());
//    print_info(" source stem positions in ");
//    // print_value("%f", toc - tic);
//    print_info(" s.\n");
//    // tic = omp_get_wtime();
//    mapping.setInputCloud(cloud_tgt->makeShared());
//    mapping.extract(cloud_pos_tgt);
//    // toc = omp_get_wtime();
//
//    // time_val += (toc - tic);
//    print_info("  (2) ");
//    print_value("%d", cloud_pos_tgt->size());
//    print_info(" target stem positions in ");
//    // print_value("%f", toc - tic);
//    print_info(" s.\n");
//
//    std::cout << "3. MATCHING STEMS:" << std::endl;
//    // tic = omp_get_wtime();
//    Eigen::Matrix4f mat_crs;
//    Matching matching;
//    matching.setPairwiseStemPositions(cloud_pos_src,
//                                      cloud_pos_tgt);
//    matching.estimateTransformation(mat_crs);
//    // toc = omp_get_wtime();
//    // time_val += (toc - tic);
//    print_info("   ");
//    print_value("%d", matching.getNumberOfMatches());
//    print_info(" pairs of correspondences in ");
//    // print_value("%f", toc - tic);
//    print_info(" s.\n");
//
//    print_info("====> [Total running time] ");
//    // print_value("%f", time_val);
//    print_info(" s.\n");
//
//    //============================ Output Matrix ==========================
//    writeTransformationMatrix(filename_result, mat_crs);

    return EXIT_SUCCESS;
}

