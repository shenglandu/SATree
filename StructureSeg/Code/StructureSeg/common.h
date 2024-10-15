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

#ifndef STRUCTURESEG_COMMON_H
#define STRUCTURESEG_COMMON_H

#include <pcl/common/common.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>


// Define point cloud
typedef pcl::PointXYZ Point3D;
typedef pcl::PointIndices Indices;
typedef pcl::PointCloud <Point3D> Cloud3D;

// Define the tree vertex and edge properties
struct GraphVertexProp{
    Point3D  coord, dir;
    std::size_t nParent;
    int idx, tree_id;
    float score;
};

struct GraphEdgeProp{
    double nWeight;
};

// Define tree graph
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS, GraphVertexProp, GraphEdgeProp> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor GraphVertexDescriptor;
typedef boost::graph_traits<Graph>::edge_descriptor GraphEdgeDescriptor;
typedef boost::graph_traits<Graph>::vertex_iterator GraphVertexIterator;
typedef boost::graph_traits<Graph>::edge_iterator GraphEdgeIterator;
typedef boost::graph_traits<Graph>::adjacency_iterator GraphAdjacencyIterator;
typedef boost::graph_traits<Graph>::out_edge_iterator  GraphOutEdgeIterator;

#endif
