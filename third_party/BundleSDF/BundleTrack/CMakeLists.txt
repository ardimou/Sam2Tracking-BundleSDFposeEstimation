# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.15)
project(BundleTrack LANGUAGES CUDA CXX C)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Enable CUDA support
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

# Set C++17 standard once
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Base flags with warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# CUDA configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --default-stream per-thread")

# Check CUDA version for architecture compatibility
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.0")
    message(WARNING "CUDA version < 11.0 may not support all specified architectures")
endif()

# Set CUDA architectures based on CUDA version
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.8")
    set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;90")
else()
    set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86")
endif()

# Feature flags
add_definitions(-DG2O=0)
add_definitions(-DTIMER=0)
add_definitions(-DPRINT_RESIDUALS_DENSE=0)
add_definitions(-DPRINT_RESIDUALS_SPARSE=0)
add_definitions(-DCUDA_RANSAC=1)
add_definitions(-DCUDA_MATCHING=1)

# RPATH settings
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find required packages with version specifications where appropriate
find_package(Boost 1.65.0 REQUIRED COMPONENTS system program_options serialization)
find_package(PCL 1.8 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV 4.0 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp 0.6 REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA 11.0 REQUIRED)
find_package(GLUT REQUIRED)
find_package(LAPACK REQUIRED)
find_package(pybind11 2.6 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(ZeroMQ QUIET)
if(NOT ZeroMQ_FOUND)
    find_library(ZMQ_LIB zmq)
    if(NOT ZMQ_LIB)
        message(FATAL_ERROR "ZeroMQ library not found")
    endif()
endif()

set(PYBIND11_CPP_STANDARD -std=c++17)

# Include directories
include_directories(
    src
    ${Boost_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${CSPARSE_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/cuda/
    ${PROJECT_SOURCE_DIR}/src/cuda/Solver/
    ${PROJECT_SOURCE_DIR}/src/Thirdparty
    ${PROJECT_SOURCE_DIR}/src/Thirdparty/g2o
)

# Optional Gurobi support
if(DEFINED GUROBI)
    message("Using Gurobi")
    add_definitions(-DGUROBI=1)
    if(NOT DEFINED GUROBI_HOME)
        message(FATAL_ERROR "GUROBI_HOME must be defined when using Gurobi")
    endif()
    include_directories("${GUROBI_HOME}/include")
    find_library(GUROBI_LIBRARY
        NAMES gurobi90 gurobi95
        PATHS "${GUROBI_HOME}/lib"
        REQUIRED
    )
    find_library(GUROBI_CXX_LIBRARY
        NAMES gurobi_c++
        PATHS "${GUROBI_HOME}/lib"
        REQUIRED
    )
endif()

# Source files
file(GLOB MY_SRC 
    ${PROJECT_SOURCE_DIR}/src/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/cuda/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/cuda/Solver/*.cpp
)

file(GLOB G2O_LIBS ${PROJECT_SOURCE_DIR}/src/Thirdparty/g2o/lib/libg2o*)

# Remove problematic PCL components if present
list(REMOVE_ITEM PCL_LIBRARIES pcl_simulation)

# CUDA files
file(GLOB CUDA_FILES
    "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/src/cuda/*.cu"
    "${PROJECT_SOURCE_DIR}/src/cuda/Solver/*.cu"
)

# Build CUDA library
add_library(MY_CUDA_LIB SHARED ${CUDA_FILES})
set_target_properties(MY_CUDA_LIB PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
target_link_libraries(MY_CUDA_LIB PUBLIC
    yaml-cpp
    ${YAML_CPP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Main library
add_library(${PROJECT_NAME} SHARED ${MY_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${YAML_CPP_LIBRARIES}
    MY_CUDA_LIB
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    ${OPENGL_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${G2O_LIBS}
    zmq
)

if(DEFINED GUROBI)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${GUROBI_LIBRARY} ${GUROBI_CXX_LIBRARY})
endif()

# Python bindings
pybind11_add_module(my_cpp 
    pybind_interface/pybind_api.cpp
    src/Frame.cpp
)
set_target_properties(my_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(my_cpp PRIVATE 
    ${PROJECT_NAME}
    MY_CUDA_LIB
    yaml-cpp
    ${YAML_CPP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${PYTHON_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    zmq
)
