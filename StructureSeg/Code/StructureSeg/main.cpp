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
#include <pcl/io/ply_io.h>
#include <pcl/console/parse.h>
#include "happly.h"

using namespace pcl::console;
using namespace happly;


int main(int argc, char** argv) {
#if defined(_OPENMP)
    print_info("[PARALLEL PROCESSING USING ");
    print_value("%d", omp_get_max_threads());
    print_info(" THREADS] \n\n");
#else
    print_info("[NON-PARALLEL PROCESSING] \n\n");
#endif

    std::string filename_source = "/mnt/data/Tree/TreeML-Data/Dataset_strstem_ply/test/2023-01-09_tum_campus_0.ply";
//    std::string filename_target = argv[2];
//    std::string filename_result = argv[3];

    // double tic, toc, time_val = 0.0;
    //============================= Load Data =============================
    std::cout << "1. LOADING RAW POINT CLOUDS:" << std::endl;
//    Cloud3D::Ptr cloud_src(new Cloud3D), cloud_tgt(new Cloud3D);
//    // tic = omp_get_wtime();
//    pcl::io::loadPLYFile<Point3D>(filename_source, *cloud_src);
//    // toc = omp_get_wtime();
//    print_info("  (1) ");
//    print_value("%d", cloud_src->size());
//    std::cout << "header: " << cloud_src->header << std::endl;
//    print_info(" source points in ");
//    // print_value("%f", toc - tic);
//    print_info(" s.\n");

    // use happly to load data
    PLYData plyIn(filename_source);
    std::vector<float> scores = plyIn.getElement("vertex").getProperty<float>("s");
    std::cout << "score size" << scores.size() << std::endl;
    for (int i = 0; i < scores.size(); i++){
        if (scores[i] > 0)
            std::cout << scores[i] << std::endl;
    }

    // tic = omp_get_wtime();
//    pcl::io::loadPLYFile<Point3D>(filename_target, *cloud_tgt);
//    // toc = omp_get_wtime();
//    print_info("  (2) ");
//    print_value("%d", cloud_tgt->size());
//    print_info(" target points in ");
//    // print_value("%f", toc - tic);
//    print_info(" s.\n");
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

    return 0;
}

