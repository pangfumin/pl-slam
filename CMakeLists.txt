project( PL-SLAM )

cmake_minimum_required(VERSION 2.7)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS regex thread system filesystem)
find_package(Eigen3   REQUIRED)
find_package(Cholmod REQUIRED)

set (G2O_INCLUDE_DIR /home/pang/software/g2o_install/include)
set (G2O_LIBRARIES
		/home/pang/software/g2o_install/lib/libg2o_cli.so
		/home/pang/software/g2o_install/lib/libg2o_core.so
		/home/pang/software/g2o_install/lib/libg2o_csparse_extension.so
		/home/pang/software/g2o_install/lib/libg2o_ext_freeglut_minimal.so
		/home/pang/software/g2o_install/lib/libg2o_hierarchical.so
		/home/pang/software/g2o_install/lib/libg2o_incremental.so
		/home/pang/software/g2o_install/lib/libg2o_interactive.so
		/home/pang/software/g2o_install/lib/libg2o_interface.so
		/home/pang/software/g2o_install/lib/libg2o_opengl_helper.so
		/home/pang/software/g2o_install/lib/libg2o_parser.so
		/home/pang/software/g2o_install/lib/libg2o_simulator.so
		/home/pang/software/g2o_install/lib/libg2o_solver_cholmod.so
		/home/pang/software/g2o_install/lib/libg2o_solver_csparse.so
		/home/pang/software/g2o_install/lib/libg2o_solver_dense.so
		/home/pang/software/g2o_install/lib/libg2o_solver_pcg.so
		/home/pang/software/g2o_install/lib/libg2o_solver_slam2d_linear.so
		/home/pang/software/g2o_install/lib/libg2o_solver_structure_only.so
		/home/pang/software/g2o_install/lib/libg2o_stuff.so
		/home/pang/software/g2o_install/lib/libg2o_types_data.so
		/home/pang/software/g2o_install/lib/libg2o_types_icp.so
		/home/pang/software/g2o_install/lib/libg2o_types_sba.so
		/home/pang/software/g2o_install/lib/libg2o_types_sclam2d.so
		/home/pang/software/g2o_install/lib/libg2o_types_sim3.so
		/home/pang/software/g2o_install/lib/libg2o_types_slam2d_addons.so
		/home/pang/software/g2o_install/lib/libg2o_types_slam2d.so
		/home/pang/software/g2o_install/lib/libg2o_types_slam3d_addons.so
		/home/pang/software/g2o_install/lib/libg2o_types_slam3d.so)



if(COMMAND cmake_policy)
  	cmake_policy(SET CMP0003 NEW)	
endif(COMMAND cmake_policy)
link_directories(${OpenCV_LIBS_DIR})
include_directories(${OpenCV2_INCLUDE_DIRS})


set(DEFAULT_HAS_MRPT OFF)
set(HAS_MRPT ${DEFAULT_HAS_MRPT} CACHE BOOL "Build the PointGrey Bumblebee2 SVO application that employs the MRPT library")

SET(BUILD_SHARED_LIBS ON)
SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -mtune=native -march=native")

add_definitions(-DBOOST_NO_CXX11_SCOPED_ENUMS)

# MRPT library (optional, only with representation purposes)
if(HAS_MRPT)
FIND_PACKAGE(MRPT REQUIRED base opengl gui hwdrivers)
set(MRPT_DONT_USE_DBG_LIBS 1) #use release libraries for linking even if "Debug" CMake build
add_definitions(-DHAS_MRPT)
endif(HAS_MRPT)


# YAML library
FIND_PACKAGE(yaml-cpp REQUIRED CONFIG PATHS ${YAML_PATHS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


# Include dirs
include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/3rdparty/DBoW2/include/
  ${PROJECT_SOURCE_DIR}/3rdparty/line_descriptor/include/
  ${G2O_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  /usr/include/suitesparse # for cholmod
)

# Set link libraries
list(APPEND LINK_LIBS 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${G2O_LIBRARIES}
		cholmod
  ${PROJECT_SOURCE_DIR}/3rdparty/DBoW2/lib/libDBoW2.so
  ${PROJECT_SOURCE_DIR}/3rdparty/line_descriptor/lib/liblinedesc.so
)

# Set source files 
if(HAS_MRPT)
list(APPEND SOURCEFILES
  src/mapHandler.cpp
  src/mapFeatures.cpp
  src/keyFrame.cpp
  src/slamConfig.cpp
  src/slamScene.cpp
)
else()
list(APPEND SOURCEFILES
  src/mapHandler.cpp
  src/mapFeatures.cpp
  src/keyFrame.cpp
  src/slamConfig.cpp
	src/auxiliar.cpp
	src/config.cpp
	src/dataset.cpp
	src/gridStructure.cpp
	src/lineIterator.cpp
	src/matching.cpp
	src/pinholeStereoCamera.cpp
	src/stereoFeatures.cpp
	src/stereoFrame.cpp
	src/stereoFrameHandler.cpp
	src/timer.cpp
)
endif()

# List all files (headers) contained by StVO-PL library
file(GLOB_RECURSE all_include_files RELATIVE "${CMAKE_SOURCE_DIR}" *.h *.hpp)

# Visualize the files of this directory in IDE creating an custom empty target
add_custom_target( plslam_includes DEPENDS ${all_include_files} SOURCES ${all_include_files} )

# Create StVO-PL library
add_library(plslam SHARED ${SOURCEFILES})

if(HAS_MRPT)
target_link_libraries(plslam ${LINK_LIBS} ${MRPT_LIBS}  )
else()
target_link_libraries(plslam ${LINK_LIBS})
endif()

# Applications 
#if(HAS_MRPT)
add_executable       ( plslam_dataset
		app/plslam_dataset.cpp
		app/file-system-tools.cpp
		app/file-system-tools.h)
target_link_libraries( plslam_dataset plslam )
#endif(HAS_MRPT)



