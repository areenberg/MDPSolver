#!/bin/bash

# Step 1: Compile the SO file in CPP_Source_Code
echo "Compiling C++ module..."
cd CPP_Source_Code || exit
g++ -O3 -fopenmp -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) *.cpp -o solvermodule$(python3-config --extension-suffix)
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful."

# Step 2: Copy the compiled .so file to Python/src/mdpsolver
echo "Copying compiled module to Python/src/mdpsolver..."
cp solvermodule$(python3-config --extension-suffix) ../Python/src/mdpsolver/
if [ $? -ne 0 ]; then
    echo "Failed to copy the module."
    exit 1
fi
echo "Copy successful."

# Step 3: Run test1.py and test2.py
echo "Running tests..."
cd ../Python/tests || exit

echo "Running test1.py..."
python3 test1.py
if [ $? -ne 0 ]; then
    echo "test1.py failed."
    exit 1
fi

echo "Running test2.py..."
python3 test2.py
if [ $? -ne 0 ]; then
    echo "test2.py failed."
    exit 1
fi

echo "All tests passed successfully."
