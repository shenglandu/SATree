cmake_minimum_required(VERSION 3.1)

#get_filename_component(MODULE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
#set(PROJECT_NAME 3rd_${MODULE_NAME})
#project(${PROJECT_NAME})

project(TETGEN)
set(package_name "TETGEN")

add_library(TETGEN STATIC
	tetgen.h
	tetgen.cxx
	predicates.cxx
)

#set_target_properties(${PROJECT_NAME} PROPERTIES
#                      FOLDER "3rd_party")


#target_include_directories(${PROJECT_NAME} PRIVATE ${EASY3D_tetgen_INCLUDE_DIR})
