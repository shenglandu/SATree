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
#include "octree_extract_clusters.h"
#include "octree_unibn.hpp"
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/common/io.h>
#include <iostream>
#include "tetgen.h"


TreeSeg::TreeSeg()
        : other_points_(new Cloud3D)
        , tree_points_(new Cloud3D)
        , db_min_pts_(50)
        , db_radius_(0.2)
        , other_id_(0)
        , stem_id_(1)
        , crown_id_(2)
        , ignore_id_(3)
        , eps_s_(0.5)
        , cut_height_(2.0)
        , radius_(0.15)
        , grid_size_(0.2)
        , scale_(1.5)
        , is_output_root_(true)
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


void TreeSeg::initialize(const std::string &config_nm) {
    // Check if the input file is ini. file
    size_t find = config_nm.find("ini");
    if (find == -1) {
        std::cout << "ini format required for configuration!" << std::endl;
        return;
    }

    // Define a property tree to load the configuration
    PT::ptree config;
    PT::ini_parser::read_ini(config_nm, config);

    // Obtain the data path
    bool is_bundle_process = config.get<bool>("Data.is_bundle_process");
    if (!is_bundle_process) {
        std::string data_path = config.get<std::string>("Data.data_path");
        // remove the
        data_path = data_path.substr(1, data_path.size() - 2);
        if (!parse_scene_name(data_path)) {
            std::cout << "fail to parse the scene data name" << std::endl;
            return;
        }
    }
    else {
        std::cout << "Currently we don't support bundle process!" << std::endl;
        return;
    }

    // Obtain the hyperparameters
    db_min_pts_ = config.get<int>("RootExtraction.db_min_pts");
    db_radius_ = config.get<float>("RootExtraction.db_radius");
    eps_s_ = config.get<float>("RootExtraction.eps_s");
    cut_height_ = config.get<float>("RootExtraction.cut_height");
    radius_ = config.get<float>("RootExtraction.radius");
    is_output_root_ = config.get<bool>("RootExtraction.is_output_root");
    grid_size_ = config.get<float>("Voxelization.grid_size");
    scale_ = config.get<float>("TreeGrouping.scale");

}


bool TreeSeg::parse_scene_name(const std::string &file_nm) {
    // Initialize
    scene_name_ = "";
    scene_path_ = "";

    // Parse the input scene file name
    std::stringstream fs(file_nm);
    std::string file_sub_nm;
    std::vector<std::string> file_sub_nms;
    char delimiter = '/';
    while (getline(fs, file_sub_nm, delimiter)) {
        file_sub_nms.push_back(file_sub_nm);
    }

    // Obtain the path and name by retrieving the tokens
    for (int i = 0; i < file_sub_nms.size(); i++) {
        if (i == file_sub_nms.size() - 1) {
            size_t find = file_sub_nms[i].find("ply");
            if (find == -1) {
                std::cout << "ply format required!" << std::endl;
                return false;
            }
            scene_name_ = file_sub_nms[i].substr(0, file_sub_nms[i].size() - 4);
        }
        else
            scene_path_ += file_sub_nms[i] + "/";
    }

    std::cout << "Working scene name: " << scene_name_ << std::endl;
    std::cout << "Working path: " << scene_path_ << std::endl;

    return true;
}


bool TreeSeg::read_clouds() {
    // Obtain the scene file name
    const std::string file_nm = scene_path_ + scene_name_ + ".ply";

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
    for (int i = 0; i < x.size(); i++) {
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
    // Check if there is input tree clouds
    if (!tree_points_){
        std::cout << "No tree points available!" << std::endl;
        return false;
    }

    // Initialize tree root containers
    roots_.clear();
    noise_idx_.clear();
    roots_idx_.clear();
    tree_root_idx_.clear();

    // Retrieve 2d stem points from the tree point cloud with the indices
    Cloud3D::Ptr proj_stem_pts(new Cloud3D);
    std::vector<int> stem_idx;
    for (int i = 0; i < tree_points_->size(); i++) {
        int sem_id = int(tree_props_[i][0]);
        if (sem_id == stem_id_){
            Point3D pi = (*tree_points_)[i];
            pi.z = 0;
            proj_stem_pts->push_back(pi);
            stem_idx.push_back(i);
        }
        // initialize the stem idx per tree point as -1
        tree_root_idx_.push_back(-1);
    }

    // Make sure there are stem points before clustering
    if(!proj_stem_pts){
        std::cout << "No stem points found!" << std::endl;
        return false;
    }

    // Perform clustering over stem points
    std::vector<Indices> proj_stem_clusters;
    OctreeEuclideanClusterExtraction<Point3D> oec;
    oec.setClusterTolerance(db_radius_);
    oec.setMinClusterSize(db_min_pts_);
    oec.setInputCloud(proj_stem_pts);
    oec.extract(proj_stem_clusters);

    // Construct 3D tree point cloud octree
    unibn::Octree<Point3D> octree_3d;
    octree_3d.initialize(*tree_points_);
    std::vector<uint32_t> neighbors_3d;

    // Construct projected tree point cloud and its octree
    Cloud3D::Ptr proj_tree_pts(new Cloud3D);
    copyPointCloud(*tree_points_, *proj_tree_pts);
    for (int i = 0; i < proj_tree_pts->size(); i++)
        (*proj_tree_pts)[i].z = 0;
    unibn::Octree<Point3D> octree_2d;
    octree_2d.initialize(*proj_tree_pts);
    std::vector<std::vector<uint32_t>> neighbors_2d(2);

    // Check the stem candidate cluster quality based on score, height, and local maxima condition
    std::vector<int> stem_ind_container;
    for (int i = 0; i < proj_stem_clusters.size(); ++i) {
        // initialize the stem candidate cluster info
        int lowest_idx, highest_idx;
        float lowest = 1000;
        float highest = -1000;
        float avg_s, max_s1 = 0.;
        Point3D centroid(0, 0, 0);
        neighbors_2d.clear();
        neighbors_3d.clear();
        stem_ind_container.clear();

        // retrieve over per point in the stem cluster
        for (int& j: proj_stem_clusters[i].indices) {
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
        }
        avg_s /= stem_ind_container.size();
        centroid.x /= stem_ind_container.size();
        centroid.y /= stem_ind_container.size();

        // check if the root is surrounded by lower tree points
        bool is_floating = false;
        octree_3d.radiusNeighbors<unibn::L2Distance<Point3D>>((*tree_points_)[lowest_idx], 2 * radius_, neighbors_3d);
        // if there is any tree point lower than the root
        for (auto& ni: neighbors_3d){
            if ((*tree_points_)[ni].z < lowest){
                is_floating = true;
                break;
            }
        }
        if (is_floating) {
            for (auto& si: stem_ind_container)
                tree_props_[si][0] = crown_id_;
            continue;
        }

        // check if the root has sufficient length
        if (highest - lowest <= 0.15 * cut_height_) {
            noise_idx_.insert(noise_idx_.end(), stem_ind_container.begin(), stem_ind_container.end());
            continue;
        }

        // check if the score is sufficiently high
        if (avg_s <= eps_s_){
            // retrieve the 2-ring spherical neighborhoods from centroid to 2d tree points
            octree_2d.radiusNeighbors<unibn::L2Distance<Point3D>>(centroid, radius_, neighbors_2d[0]);
            octree_2d.radiusNeighbors<unibn::L2Distance<Point3D>>(centroid, 3 * radius_, neighbors_2d[1]);
            if (neighbors_2d[0].empty() or neighbors_2d[1].empty()){
                noise_idx_.insert(noise_idx_.end(), stem_ind_container.begin(), stem_ind_container.end());
                continue;
            }
            // collect scores of the 1st neighbourhood and obtain the max score value
            std::vector<float> neigh_s1, neigh_s2;
            for (auto& ni: neighbors_2d[0]){
                if ((*tree_points_)[ni].z - highest <= 0.5)
                    neigh_s1.push_back(tree_props_[ni][1]);
            }
            float neigh_max_s1 = *std::max_element(neigh_s1.begin(), neigh_s1.end());
            max_s1 = std::max(neigh_max_s1, max_s1);
            // check if the 2nd neighbour contains crown points, collect scores in the 2nd neighbour
            int num_crown = 0;
            for (auto& ni: neighbors_2d[1]){
                if ((*tree_points_)[ni].z - highest <= 0.5){
                    neigh_s2.push_back(tree_props_[ni][1]);
                    int sem_id = int(tree_props_[ni][0]);
                    if (sem_id == crown_id_)
                        num_crown ++;
                }
            }
            // if the neighbourhood contains no crown, then the stem is invalid
            if (num_crown == 0){
                noise_idx_.insert(noise_idx_.end(), stem_ind_container.begin(), stem_ind_container.end());
                continue;
            }
            // if any score in 2nd neighbor is greater than local maximal, then the stem is invalid
            float max_s2 = *std::max_element(neigh_s2.begin(), neigh_s2.end());
            float neigh_avg_s1 = std::accumulate(neigh_s1.begin(), neigh_s1.end(), 0.0) / neigh_s1.size();
            if (neigh_avg_s1 <= eps_s_){
                if (max_s1 <= 0.2 or max_s2 > max_s1){
                    noise_idx_.insert(noise_idx_.end(), stem_ind_container.begin(), stem_ind_container.end());
                    continue;
                }
            }
        }

        // obtain the root location by stem points below the cutting height
        Point3D root(0, 0, lowest);
        int n_pts = 0;
        for (int& j: proj_stem_clusters[i].indices){
            int stem_ind = stem_idx[j];
            float zj = (*tree_points_)[stem_ind].z;
            if (zj - lowest <= cut_height_){
                root.x += (*proj_stem_pts)[j].x;
                root.y += (*proj_stem_pts)[j].y;
                n_pts ++;
            }
        }
        root.x /= float(n_pts);
        root.y /= float(n_pts);
        roots_.push_back(root);
        roots_idx_.push_back(stem_ind_container);
    }

    // Filter the roots based on height
    filter_roots();

    // Assign root indices to tree points
    for (int root_i = 0; root_i < roots_idx_.size(); root_i ++) {
        for (int& j: roots_idx_[root_i])
            tree_root_idx_[j] = root_i;
    }

    // Clear temporary data and return
    std::sort(noise_idx_.begin(), noise_idx_.end());
    proj_stem_pts->clear();
    proj_tree_pts->clear();
    octree_2d.clear();
    octree_3d.clear();
    std::cout << roots_.size() << " number of roots have been extracted" << std::endl;
    std::cout << noise_idx_.size() << " points are detected as noises" << std::endl;

    // Optionally output root positions
    if (is_output_root_)
        output_root_xyz();

    return true;
}


bool TreeSeg::group_trees() {
    // Check if there are detected roots
    if (roots_.empty()) {
        std::cout << "No tree roots detected!" << std::endl;
        return false;
    }

    // Voxelize the input tree points
    std::cout << "Voxelizing tree points..." << std::endl;
    voxelize_tree_points();

    // Group individual tree points by graph shortest path
    std::cout << "Building delaunay..." << std::endl;
    build_delaunay();

    std::cout << "Extracting MST..." << std::endl;
    extract_mst();

    std::cout << "Assign tree id..." << std::endl;
    assign_tree_id();

    return true;
}


void TreeSeg::output_tree_seg() {
    // Initialize containers
    std::vector<float> x, y, z;
    std::vector<int> ins, roots;

    // Retrive over graph vertices
    std::pair<GraphVertexIterator, GraphVertexIterator> vp = vertices(MST_);
    for (GraphVertexIterator vIter = vp.first; vIter != vp.second; ++vIter) {
        GraphVertexDescriptor vi = *vIter;
        if (degree(vi, MST_) != 0 ) { // ignore isolated vertices
            int tree_id = MST_[vi].tree_id;
            int voxel_id = MST_[vi].idx;
            if (voxel_id < 0) continue;
            // retrieve raw points in current voxel
            for (int& j: voxel_idx_[voxel_id]) {
                x.push_back((*tree_points_)[j].x);
                y.push_back((*tree_points_)[j].y);
                z.push_back((*tree_points_)[j].z);
                ins.push_back(tree_id);
                roots.push_back(tree_root_idx_[j]);
            }
        }
    }

    // Organize the data
    int nPts = x.size();
    PLYData plyOut;
    plyOut.addElement("vertex", nPts);
    plyOut.getElement("vertex").addProperty<float>("x", x);
    plyOut.getElement("vertex").addProperty<float>("y", y);
    plyOut.getElement("vertex").addProperty<float>("z", z);
    plyOut.getElement("vertex").addProperty<int>("ins", ins);
    plyOut.getElement("vertex").addProperty<int>("r", roots);

    // Write to ply
    const std::string file_nm = scene_path_ + scene_name_ + "_seg.ply";
    plyOut.write(file_nm, DataFormat::Binary);

}


void TreeSeg::output_root_xyz() {
    // Check if there are detected roots
    if (roots_.empty()) {
        std::cout << "No tree roots detected!" << std::endl;
        return;
    }

    // Write root coordinates to the file
    const std::string file_nm = scene_path_ + scene_name_ + "_root.xyz";
    std::ofstream root_file;
    root_file.open(file_nm);
    for (int i = 0; i < roots_.size(); i++)
        root_file << roots_[i].x << " " << roots_[i].y << " " << roots_[i].z << std::endl;
    root_file.close();

}


void TreeSeg::voxelize_tree_points() {
    // Check if there is input tree clouds
    if (!tree_points_){
        std::cout << "No tree points available!" << std::endl;
        return;
    }

    // Initialize
    voxel_idx_.clear();
    tree_voxel_map_idx_.clear();
    for (int i = 0; i < tree_points_->size(); i++)
        tree_voxel_map_idx_.push_back(int(-100));

    // Mark noises in the tree points as -1
    if (!noise_idx_.empty()) {
        for (int& ni: noise_idx_)
            tree_voxel_map_idx_[ni] = -1;
    }

    // Voxelize points using octree
    pcl::octree::OctreePointCloudPointVector<Point3D> oct(grid_size_);
    oct.setInputCloud(tree_points_);
    oct.addPointsFromInputCloud();

    // Traverse the octree leafs and store the indices
    std::vector<int> raw_idx, denoised_idx;
    int voxel_id = 0;
    for (auto it = oct.leaf_depth_begin(); it != oct.leaf_depth_end(); ++it) {
        raw_idx.clear();
        denoised_idx.clear();
        auto leaf = it.getLeafContainer();
        leaf.getPointIndices(raw_idx);
        for (int& j: raw_idx) {
            // check if the tree point is marked as noise
            if (tree_voxel_map_idx_[j] != -1) {
                tree_voxel_map_idx_[j] = voxel_id;
                denoised_idx.push_back(j);
            }
        }
        if (!denoised_idx.empty()) {
            voxel_idx_.push_back(denoised_idx);
            voxel_id ++;
        }
    }
    std::cout << tree_points_->size() << " points have been down-sampled to " <<  voxel_idx_.size() << std::endl;

}


void TreeSeg::filter_roots() {
    // Check if there are detected roots
    if (roots_.empty()) {
        std::cout << "No tree roots detected!" << std::endl;
        return;
    }

    // Obtain the averaged root height and median root height
    int nRoots = roots_.size();
    std::vector<float> root_heights;
    for (auto ri: roots_)
        root_heights.push_back(ri.z);
    std::sort(root_heights.begin(), root_heights.end());
    float avg_h = std::accumulate(root_heights.begin(), root_heights.end(), 0.0) / nRoots;
    float median_h;
    if (nRoots % 2 != 0)
        median_h = root_heights[nRoots / 2];
    else
        median_h = (root_heights[(nRoots - 1) / 2] + root_heights[nRoots / 2]) / 2.0;
    float thres_h = (avg_h + median_h) / 2.0;

    // Retrieve roots and filter out the root that is too high above the average
    for (int i = roots_.size() - 1; i >= 0; -- i) {
        if (roots_[i].z - thres_h > cut_height_) {
            roots_.erase(roots_.begin() + i);
            roots_idx_.erase(roots_idx_.begin() + i);
        }
    }
}


void TreeSeg::build_delaunay() {
    // Initialize
    delaunay_.clear();

    // Check if the tree points have been denoised and voxelized
    if (voxel_idx_.empty()) {
        std::cout << "Tree points need to be voxelized first!" << std::endl;
        return;
    }

    // Read voxelized points as vertices into the graph
    int n_pts = 0;
    for (int i = 0; i < voxel_idx_.size(); i++) {
        // Obtain the centralized points with its properties
        Point3D centroid(0, 0, 0), direction(0, 0, 0);
        float score = 0.;
        int sem_id, root_id = -1;
        int stem_count, crown_count = 0;
        std::unordered_map<int, int> root_id_container;
        for (int& j: voxel_idx_[i]) {
            // accumulate centroid
            centroid.x += (*tree_points_)[j].x;
            centroid.y += (*tree_points_)[j].y;
            centroid.z += (*tree_points_)[j].z;
            // accumulate direction
            direction.x += tree_props_[j][2];
            direction.y += tree_props_[j][3];
            direction.z += tree_props_[j][4];
            // accumulate score
            score += tree_props_[j][1];
            // accumulate stem count and crown count
            if (tree_props_[j][0] == stem_id_) {
                root_id_container[tree_root_idx_[j]] ++;
                stem_count++;
            }
            else
                crown_count++;
        }
        // average the coordinates and the score
        centroid.x /= voxel_idx_[i].size();
        centroid.y /= voxel_idx_[i].size();
        centroid.z /= voxel_idx_[i].size();
        direction.x /= voxel_idx_[i].size();
        direction.y /= voxel_idx_[i].size();
        direction.z /= voxel_idx_[i].size();
        score /= voxel_idx_[i].size();
        if (stem_count > crown_count) {
            sem_id = stem_id_;
            // assign the most frequent root id as the voxel root id
            int max_count = 0;
            for (auto ri : root_id_container) {
                if (max_count < ri.second) {
                    root_id = ri.first;
                    max_count = ri.second;
                }
            }
        }
        else
            sem_id = crown_id_;

        // read the point information into the vertex
        GraphVertexProp vi;
        vi.coord = centroid;
        vi.dir = direction;
        vi.nParent = 0;
        vi.score = score;
        vi.idx = i;
        vi.sem_id = sem_id;
        vi.root_id = root_id;
        vi.tree_id = -100;
        // shift the coordinate if point is crown
        if (vi.sem_id == stem_id_)
            vi.shifted_coord = vi.coord;
        else
            vi.shifted_coord = shift_point_3d(centroid, direction, score);
        // add vertex to the graph
        add_vertex(vi, delaunay_);
        n_pts ++;
    }

    // Read roots into the graph and generate the lowest pseudo root
    pseudo_root_ = Point3D(0, 0, 1000);
    for (int i = 0; i < roots_.size(); i++) {
        // read roots into the graph
        GraphVertexProp vi;
        vi.coord = roots_[i];
        vi.dir = Point3D(0, 0, 0);
        vi.shifted_coord = vi.coord;
        vi.nParent = 0;
        vi.score = 1.;
        vi.idx = -1;  // -1 means the vertex is pseudo added
        vi.sem_id = stem_id_;
        vi.root_id = i;
        vi.tree_id = -100;
        add_vertex(vi, delaunay_);
        // accumulate to pseudo root
        pseudo_root_.x += roots_[i].x;
        pseudo_root_.y += roots_[i].y;
        if (pseudo_root_.z > roots_[i].z)
            pseudo_root_.z = roots_[i].z;
        // increase point count
        n_pts ++;
    }
    pseudo_root_.x /= roots_.size();
    pseudo_root_.y /= roots_.size();
    pseudo_root_.z -= 20;

    // Add pseudo root to the graph
    GraphVertexProp vr;
    vr.coord = pseudo_root_;
    vr.dir = Point3D(0, 0, 0);
    vr.shifted_coord = vr.coord;
    vr.nParent = 0;
    vr.score = 1.;
    vr.idx = -1;  // -1 means the vertex is pseudo added
    vr.sem_id = stem_id_;
    vr.root_id = -1;
    vr.tree_id = -100;
    add_vertex(vr, delaunay_);
    n_pts += 1;

    // Construct delaunay edges
    tetgenio tet_in, tet_out;
    tet_in.numberofpoints = n_pts;
    tet_in.pointlist = new REAL[tet_in.numberofpoints * 3];
    int count = 0;
    std::pair<GraphVertexIterator, GraphVertexIterator> vp = vertices(delaunay_);
    for (GraphVertexIterator vIter = vp.first; vIter != vp.second; ++vIter) {
        Point3D pi = (delaunay_)[*vIter].coord;
        tet_in.pointlist[count * 3 + 0] = pi.x;
        tet_in.pointlist[count * 3 + 1] = pi.y;
        tet_in.pointlist[count * 3 + 2] = pi.z;
        ++count;
    }
    const std::string str("Q");
    tetrahedralize(const_cast<char*>(str.c_str()), &tet_in, &tet_out);
    for (long nTet = 0; nTet < tet_out.numberoftetrahedra; nTet++) {
        long tet_first = nTet * tet_out.numberofcorners;
        for (long i = tet_first; i < tet_first + tet_out.numberofcorners; i++)
            for (long j = i + 1; j < tet_first + tet_out.numberofcorners; j++)
                add_edge(vertex(tet_out.tetrahedronlist[i], delaunay_), vertex(tet_out.tetrahedronlist[j], delaunay_), delaunay_);
    }

    std::cout << "V: " << num_vertices(delaunay_) << std::endl;
    std::cout << "E: " << num_edges(delaunay_) << std::endl;

    // Obtain root vertex in the graph
    obtain_root_vertex();

    // Compute graph weights
    compute_graph_weights();

}


void TreeSeg::extract_mst() {
    // Initialize
    MST_.clear();

    // Add vertices to MST
    std::pair<GraphVertexIterator, GraphVertexIterator> vp = vertices(delaunay_);
    for (GraphVertexIterator vIter = vp.first; vIter != vp.second; ++vIter) {
        GraphVertexProp vi;
        vi.coord = (delaunay_)[*vIter].coord;
        vi.nParent = (delaunay_)[*vIter].nParent;
        vi.idx = (delaunay_)[*vIter].idx;
        vi.sem_id = (delaunay_)[*vIter].sem_id;
        vi.root_id = (delaunay_)[*vIter].root_id;
        vi.tree_id = (delaunay_)[*vIter].tree_id;
        add_vertex(vi, MST_);
    }

    // Obtain MST edges by shortest path
    std::vector<double> dists(num_vertices(delaunay_));
    std::vector<GraphVertexDescriptor> vecParent(num_vertices(delaunay_));
    dijkstra_shortest_paths(delaunay_, pseudo_root_vertex_, weight_map(get(&GraphEdgeProp::nWeight, delaunay_))
            .distance_map(&dists[0])
            .predecessor_map(&(vecParent[0])));
    for (unsigned int i = 0; i < vecParent.size(); ++i) {
        if (vertex(i, MST_) != vecParent.at(i)) {
            GraphEdgeProp ei;
            ei.nWeight = 0.0;
            add_edge(vertex(i, MST_), vertex(vecParent.at(i), MST_), ei, MST_);
        }
        MST_[vertex(i, MST_)].nParent = vecParent.at(i);
    }

    std::cout << "V: " << num_vertices(MST_) << std::endl;
    std::cout << "E: " << num_edges(MST_) << std::endl;

}


void TreeSeg::assign_tree_id() {
    // Remote pseudo root
    clear_vertex(pseudo_root_vertex_, MST_);

    // Retrieve vertices in MST and assign tree id
    for (int i = 0; i < root_vertices_.size(); i++) {
        GraphVertexDescriptor ri = root_vertices_[i];
        std::vector<GraphVertexDescriptor> stack, stack_container;
        stack.push_back(ri);
        int tree_count, crown_count = 0;
        while (true) {
            GraphVertexDescriptor vi = stack.back();
            (MST_)[vi].tree_id = i;
            tree_count ++;
            if ((MST_)[vi].sem_id == crown_id_) crown_count++;
            stack.pop_back();
            stack_container.push_back(vi);
            std::pair<GraphAdjacencyIterator, GraphAdjacencyIterator> aj = adjacent_vertices(vi, MST_);
            for (GraphAdjacencyIterator aIter = aj.first; aIter != aj.second; ++aIter) {
                if (*aIter != (MST_)[vi].nParent) {
                        stack.push_back(*aIter);
                }
            }
            if (stack.empty()) {
//                if (crown_count == 0) {
//                    for (auto vj: stack_container)
//                        (MST_)[vj].tree_id = -100;
//                }
                break;
            }
        }
    }

}


void TreeSeg::compute_graph_weights() {
    // Initialize
    std::pair<GraphEdgeIterator, GraphEdgeIterator> ep = edges(delaunay_);
    GraphVertexDescriptor vs, vt;
    Point3D ps, pt, ds, dt;

    // Retrieve over edges
    for (GraphEdgeIterator eIter = ep.first; eIter != ep.second; ++eIter) {
        // obtain coordinates of source and target vertex
        vs = source(*eIter, delaunay_);
        vt = target(*eIter, delaunay_);
        double dist = compute_pair_distance((delaunay_)[vs].coord, (delaunay_)[vt].coord);
        double shifted_dist = compute_pair_distance((delaunay_)[vs].shifted_coord, (delaunay_)[vt].shifted_coord);
        double s = ((delaunay_)[vs].score + (delaunay_)[vt].score) / 2.0;
//        // check the pairwise distance if both vertices are stem
        if ((delaunay_)[vs].sem_id == stem_id_ and (delaunay_[vt].sem_id == stem_id_)) {
            if ((delaunay_)[vs].root_id == (delaunay_)[vt].root_id) {
                s = std::max(0., 1. - s);
                (delaunay_)[*eIter].nWeight = s * dist;
                continue;
            }
        }
        // set the weight as the shifted distance between source and target
        (delaunay_)[*eIter].nWeight = shifted_dist;
    }

    // Assign pseudo edges between pseudo root and roots with weight 0
    for (int i = 0; i < root_vertices_.size(); i++){
        GraphVertexDescriptor ri = root_vertices_[i];
        if (!edge(pseudo_root_vertex_, ri, delaunay_).second){
            GraphEdgeProp ei;
            ei.nWeight = 0.0;
            add_edge(pseudo_root_vertex_, ri, ei, delaunay_);
        }
        else
            (delaunay_)[edge(pseudo_root_vertex_, ri, delaunay_).first].nWeight = 0.0;
    }

}


void TreeSeg::obtain_root_vertex() {
    // Obtain the root vertices and pseudo root vertex
    std::pair<GraphVertexIterator, GraphVertexIterator> vp = vertices(delaunay_);
    for (GraphVertexIterator vIter = vp.first; vIter != vp.second; ++vIter)
    {
        Point3D pi = (delaunay_)[*vIter].coord;
        double dist_2_pseudo = compute_pair_distance(pi, pseudo_root_);
        if (dist_2_pseudo <= 0.0001) {
            pseudo_root_vertex_ = *vIter;
            continue;
        }
        for (auto& ri: roots_) {
            double dist_2_ri = compute_pair_distance(pi, ri);
            if (dist_2_ri <= 0.0001)
            {
                GraphVertexDescriptor vr = *vIter;
                root_vertices_.push_back(vr);
            }
        }
    }

}


Point3D TreeSeg::shift_point_3d(Point3D p, Point3D dir, float s) {
    // Normailze the direction
    dir = normalize_point_3d(dir);

    // Inverse the score
    s = std::max(1. - s, 0.);

    // Fix the scale with minimum 2d projection
    float scale = 1000.;
    Point3D p_2_r;
    for (auto& ri: roots_) {
        p_2_r.getArray3fMap() = ri.getArray3fMap() - p.getArray3fMap();
        if (p_2_r.x * dir.x > 0 and p_2_r.y * dir.y > 0) {
            float proj = compute_pair_distance_2d(p, ri);
            if (scale > proj)
                scale = proj;
        }
    }
    if (scale == 1000.)
        scale = scale_;

    // Shift the coordinate
    p.getArray3fMap() += scale * s * dir.getArray3fMap();

    return p;
}


Point3D TreeSeg::normalize_point_3d(Point3D p) {
    // Initialize the origin
    Point3D po(0, 0, 0);

    // Compute the distance
    double dist = compute_pair_distance(p, po) + 1e-5;

    // Normalize the dimensions accordingly
    p.x /= dist;
    p.y /= dist;
    p.z /= dist;

    return p;
}


double TreeSeg::compute_pair_distance(Point3D p1, Point3D p2) {
    // Compute dx, dy, dz
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;

    // Obtain distance
    double dist = sqrt(dx * dx + dy * dy + dz * dz);
    return dist;
}


double TreeSeg::compute_pair_distance_2d(Point3D p1, Point3D p2) {
    // Compute dx, dy, dz
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;

    // Obtain distance
    double dist = sqrt(dx * dx + dy * dy);
    return dist;
}