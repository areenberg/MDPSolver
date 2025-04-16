@echo off
setlocal enabledelayedexpansion

REM Step 1: Navigate to CPP_Source_Code
cd /d "%~dp0CPP_Source_Code"

REM Get the Python version (e.g., 3.11)
for /f %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i

REM Write the CMakeLists.txt
echo Writing CMakeLists.txt...
(
echo cmake_minimum_required(VERSION !PYTHON_VERSION!)
echo.
echo project(solvermodule)
echo.
echo add_subdirectory(pybind11)
echo.
echo file(GLOB SOURCES "*.cpp")
echo.
echo pybind11_add_module(solvermodule %%SOURCES%%)
echo.
echo set_target_properties(solvermodule PROPERTIES CXX_STANDARD 11)
echo.
echo find_package(OpenMP)
echo if(OpenMP_CXX_FOUND^)
echo ^    target_compile_options(solvermodule PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/openmp:llvm>")
echo ^    target_link_libraries(solvermodule PUBLIC OpenMP::OpenMP_CXX)
echo endif()
) > CMakeLists.txt

REM Step 2: Run cmake build commands
echo Creating build directory and compiling...
if not exist build mkdir build
cd build
cmake ..
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Build failed.
    exit /b %errorlevel%
)

REM Step 3: Copy compiled PYD to Python/src/mdpsolver
cd Release
for %%f in (solvermodule*.pyd) do (
    copy /Y %%f ..\..\Python\src\mdpsolver\
    echo Copied %%f to Python\src\mdpsolver
)

REM Step 4: Run tests
cd ..\..\Python\tests
echo Running test1.py...
python test1.py
if %errorlevel% neq 0 (
    echo test1.py failed.
    exit /b %errorlevel%
)

echo Running test2.py...
python test2.py
if %errorlevel% neq 0 (
    echo test2.py failed.
    exit /b %errorlevel%
)

echo All tests completed successfully.
