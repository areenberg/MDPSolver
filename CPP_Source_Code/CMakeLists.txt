cmake_minimum_required(VERSION 3.13)

project(solvermodule)

add_subdirectory(pybind11)

file(GLOB SOURCES "*.cpp")

pybind11_add_module(solvermodule ${SOURCES})

set_target_properties(solvermodule PROPERTIES CXX_STANDARD 11)

# Find OpenMP package
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    # Set compiler options for LLVM OpenMP
    target_compile_options(solvermodule PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/openmp:llvm>")
    # Link OpenMP flags to the target
    target_link_libraries(solvermodule PUBLIC OpenMP::OpenMP_CXX)
endif()