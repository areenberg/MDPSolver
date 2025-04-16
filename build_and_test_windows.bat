@echo off
setlocal enabledelayedexpansion

REM Go to CPP_Source_Code
cd /d "%~dp0CPP_Source_Code" || (
    echo Failed to change directory to CPP_Source_Code
    exit /b 1
)

REM Make sure the file exists
if not exist CMakeLists.txt (
    REM Get Python version like 3.11
    for /f %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
    REM Create CMakeLists.txt
    echo cmake_minimum_required(VERSION !PYTHON_VERSION!) > CMakeLists.txt
    echo. >> CMakeLists.txt
    echo project(solvermodule) >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo add_subdirectory(pybind11) >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo file(GLOB SOURCES "*.cpp") >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo pybind11_add_module(solvermodule ^${SOURCES^}) >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo set_target_properties(solvermodule PROPERTIES CXX_STANDARD 11) >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo find_package(OpenMP) >> CMakeLists.txt
    echo if(OpenMP_CXX_FOUND) >> CMakeLists.txt
    echo     target_compile_options(solvermodule PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/openmp:llvm>") >> CMakeLists.txt
    echo     target_link_libraries(solvermodule PUBLIC OpenMP::OpenMP_CXX) >> CMakeLists.txt
    echo endif() >> CMakeLists.txt
)

REM Run cmake
if not exist build mkdir build
cd build

echo Running CMake...
cmake .. || (
    echo CMake configuration failed.
    exit /b 1
)

echo Building project...
cmake --build . --config Release || (
    echo Build failed.
    exit /b 1
)

REM Copy PYD-file to Python/src/mdpsolver
cd Release
for %%f in (solvermodule*.pyd) do (
    copy /Y %%f ..\..\..\Python\src\mdpsolver\
    echo Copied %%f to Python\src\mdpsolver
)

REM Run tests
cd ..\..\Python\tests
echo Running test1.py...
python test1.py || (
    echo test1.py failed.
    exit /b 1
)

echo Running test2.py...
python test2.py || (
    echo test2.py failed.
    exit /b 1
)

echo All tests completed successfully.
pause
