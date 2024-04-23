cmake_minimum_required(VERSION 3.5)
project(mdpsolver)

add_subdirectory(pybind11)

file(GLOB SOURCES "*.cpp")

pybind11_add_module(mdpsolver ${SOURCES})

set_target_properties(mdpsolver PROPERTIES CXX_STANDARD 11)