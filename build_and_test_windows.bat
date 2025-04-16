@echo off
setlocal enabledelayedexpansion

REM Go to CPP_Source_Code
cd /d "%~dp0CPP_Source_Code" || (
    echo Failed to change directory to CPP_Source_Code
    exit /b 1
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
cd ..
cd ..
cd ..
echo Running test1.py...
python Python/tests/test1.py || (
    echo test1.py failed.
    exit /b 1
)

echo Running test2.py...
python Python/tests/test2.py || (
    echo test2.py failed.
    exit /b 1
)

echo All tests completed successfully.
pause
