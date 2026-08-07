cmake_minimum_required(VERSION 3.15)

project(solvermodule LANGUAGES CXX)

# pybind11 now comes from PyPI (declared as a build-time dependency in
# Python/pyproject.toml) instead of being vendored as a subdirectory.
find_package(pybind11 CONFIG REQUIRED)

file(GLOB SOURCES "*.cpp")

pybind11_add_module(solvermodule ${SOURCES})

set_target_properties(solvermodule PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)

# Find OpenMP package
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    # Set compiler options for LLVM OpenMP
    target_compile_options(solvermodule PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/openmp:llvm>")
    # Link OpenMP flags to the target
    target_link_libraries(solvermodule PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install the compiled extension into the mdpsolver package directory of the wheel.
install(TARGETS solvermodule DESTINATION mdpsolver)