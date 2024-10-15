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
#include "voxel_grid_fix.h"
#include "octree_extract_clusters.h"
#include "octree_unibn.hpp"
#include <pcl/octree/octree_pointcloud_pointvector.h>
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
        , eps_dist_(3.0)
        , cut_height_(2.0)
        , radius_(0.15)
        , grid_size_(0.2)
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
        if (is_floating) continue;

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
                if (max_s1 <= 0.5 * eps_s_ or max_s2 > max_s1){
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
    }

    // Clear temporary data and return
    std::sort(noise_idx_.begin(), noise_idx_.end());
    proj_stem_pts->clear();
    proj_tree_pts->clear();
    octree_2d.clear();
    octree_3d.clear();
    std::cout << roots_.size() << " number of roots have been extracted" << std::endl;
    std::cout << noise_idx_.size() << " points are detected as noises" << std::endl;

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


void TreeSeg::output_tree_seg(const std::string &file_nm) {
    // Initialize containers
    std::vector<float> x, y, z;
    std::vector<int> ins;

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
            }
        }
    }

    // Write data to output ply
    int nPts = x.size();
    PLYData plyOut;
    plyOut.addElement("vertex", nPts);
    plyOut.getElement("vertex").addProperty<float>("x", x);
    plyOut.getElement("vertex").addProperty<float>("y", y);
    plyOut.getElement("vertex").addProperty<float>("z", z);
    plyOut.getElement("vertex").addProperty<int>("ins", ins);
    plyOut.write(file_nm, DataFormat::Binary);

}


void TreeSeg::output_root_xyz(const std::string &file_nm) {
    // Check if there are detected roots
    if (roots_.empty()) {
        std::cout << "No tree roots detected!" << std::endl;
        return;
    }

    // Write root coordinates to the file
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
        }
        // average the coordinates and the score
        centroid.x /= voxel_idx_[i].size();
        centroid.y /= voxel_idx_[i].size();
        centroid.z /= voxel_idx_[i].size();
        direction.x /= voxel_idx_[i].size();
        direction.y /= voxel_idx_[i].size();
        direction.z /= voxel_idx_[i].size();
        score /= voxel_idx_[i].size();

        // read the point information into the vertex
        GraphVertexProp vi;
        vi.coord = centroid;
        vi.dir = direction;
        vi.nParent = 0;
        vi.score = score;
        vi.idx = i;
        vi.tree_id = -100;
        add_vertex(vi, delaunay_);
        n_pts ++;
    }

    // Read roots into the graph and generate the lowest pseudo root
    pseudo_root_ = Point3D(0, 0, 1000);
    for (auto& ri: roots_) {
        // read roots into the graph
        GraphVertexProp vi;
        vi.coord = ri;
        vi.dir = Point3D(0, 0, 0);
        vi.nParent = 0;
        vi.score = 1.;
        vi.idx = -1;  // -1 means the vertex is pseudo added
        vi.tree_id = -100;
        add_vertex(vi, delaunay_);
        // accumulate to pseudo root
        pseudo_root_.x += ri.x;
        pseudo_root_.y += ri.y;
        if (pseudo_root_.z > ri.z)
            pseudo_root_.z = ri.z;
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
    vr.nParent = 0;
    vr.score = 1.;
    vr.idx = -1;  // -1 means the vertex is pseudo added
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
        std::vector<GraphVertexDescriptor> stack;
        stack.push_back(ri);
        while (true) {
            GraphVertexDescriptor vi = stack.back();
            (MST_)[vi].tree_id = i;
            stack.pop_back();
            std::pair<GraphAdjacencyIterator, GraphAdjacencyIterator> aj = adjacent_vertices(vi, MST_);
            for (GraphAdjacencyIterator aIter = aj.first; aIter != aj.second; ++aIter) {
                if (*aIter != (MST_)[vi].nParent) {
                    Point3D ps = (MST_)[vi].coord;
                    Point3D pt = (MST_)[*aIter].coord;
//                    float dist = pcl::euclideanDistance(ps, pt);
//                    if (dist <= 3.0)  stack.push_back(*aIter);
                    stack.push_back(*aIter);
                }
            }
            if (stack.empty())
                break;
        }
    }
}


void TreeSeg::compute_graph_weights() {
    // Initialize
    std::pair<GraphEdgeIterator, GraphEdgeIterator> ep = edges(delaunay_);
    GraphVertexDescriptor vs, vt;
    Point3D ps, pt, ds, dt;
    float s1, s2;

    // Retrieve over edges
    for (GraphEdgeIterator eIter = ep.first; eIter != ep.second; ++eIter) {
        // obtain coordinates of source and target vertex
        vs = source(*eIter, delaunay_);
        vt = target(*eIter, delaunay_);
        ps = (delaunay_)[vs].coord;
        pt = (delaunay_)[vt].coord;
        // obtain the score and direction information
        ds = (delaunay_)[vs].dir;
        dt = (delaunay_)[vt].dir;
        s1 = 1.0 - (delaunay_)[vs].score;
        s2 = 1.0 - (delaunay_)[vt].score;
        if (s1 < 0) s1 = 0.;
        if (s2 < 0) s2 = 0.;
        // normalize the directions
        ds = normalize_point_3d(ds);
        dt = normalize_point_3d(dt);
        // shift the original coordinates based on scaled offsets
        ps.getArray3fMap() = ps.getArray3fMap() + 1.5 * s1 * ds.getArray3fMap();
        pt.getArray3fMap() = pt.getArray3fMap() + 1.5 * s2 * dt.getArray3fMap();
        // set the weight as the shifted distance between source and target
        (delaunay_)[*eIter].nWeight = compute_pair_distance(ps, pt);
    }

    // Assign pseudo edges between pseudo root and roots with weight 0
    for (int i = 0; i < root_vertices_.size(); i++){
        GraphVertexDescriptor ri = root_vertices_[i];
        if (!edge(pseudo_root_vertex_, ri, delaunay_).second)  {
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