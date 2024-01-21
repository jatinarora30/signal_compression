cmake_minimum_required(VERSION 3.10)

project(Compression)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS} include)

add_executable(Compression src/Compression.cpp)

