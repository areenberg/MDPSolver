cmake_minimum_required(VERSION 3.5)
project(solvermodule)

add_subdirectory(pybind11)

file(GLOB SOURCES "*.cpp")

pybind11_add_module(solvermodule ${SOURCES})

set_target_properties(solvermodule PROPERTIES CXX_STANDARD 11)