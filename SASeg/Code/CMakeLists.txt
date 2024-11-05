cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(SASeg)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find packages vtk
find_package(VTK REQUIRED NO_MODULE)

# Find packages pcl
find_package(PCL REQUIRED COMPONENTS
        common io features filters segmentation
        sample_consensus kdtree registration
        CONFIG
        )

# FInd packages openmp
find_package(OpenMP)
if (OpenMP_FOUND)
    # Add OpenMP flags to all targets
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_libraries(${OpenMP_CXX_FLAGS})
endif ()

# Manually set path pf boost and find the package
SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "/mnt/software/boost_1_78_0")
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/mnt/software/boost_1_78_0/libs")
# It's "Boost", not "BOOST" or "boost". Case matters.
find_package(Boost REQUIRED)

# Add subdirectory for the tetgen library
add_subdirectory(3rd_party/tetgen)

# Set project sources
set(PROJECT_SOURCES
        3rd_party/octree/octree_extract_clusters.cpp
        SASeg/main.cpp
        SASeg/tree_seg.cpp
        )

# Add executable target
add_executable(SASeg ${PROJECT_SOURCES})

# Add include directories to target
target_include_directories(SASeg PUBLIC
        ${PCL_INCLUDE_DIRS} # PCL include directories
        ${Boost_INCLUDE_DIRS} # Boost include directories
        ${Boost_headers_DIR} # Boost header directories
        3rd_party/ply
        3rd_party/tetgen
        3rd_party/octree
        )

# Link libraries to target
target_link_libraries(SASeg PUBLIC
        TETGEN # TETGEN Library
        ${PCL_LIBRARIES} # PCL libraries
        ${VTK_LIBRARIES} # VTK libraries
        ${Boost_LIBRARIES} # Boost libraries
        )
