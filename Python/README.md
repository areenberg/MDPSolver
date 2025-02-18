# MDPSolver

<img src="https://github.com/areenberg/MDPSolver/blob/anders_development/Images/MDPSolver_logo_3.png" alt="Logo" width="180">

MDPSolver is a Python package for large Markov Decision Processes (MDPs) with infinite-horizons. 

## Features

* Fast solver: Our C++-based solver is substantially faster than other MDP packages available for Python. See details in the documentation.
* Two optimality criteria: *Discounted* and *Average* reward.
* Three optimization algorithms: *Value iteration*, *Policy iteration*, and *Modified policy iteration*.
* Three value-update methods: *Standard*, *Gauss–Seidel*, and *Successive over-relaxation*.
* Supports sparse matrices.
* Employs parallel computing.

# Installation

## Linux

Install directly from PyPI with:

```
pip install mdpsolver
```

MDPSolver works *out of the box* on Ubuntu 22 and newer.

### GLIBC not found

Some users will encounter the `version 'GLIBC_2.32' not found` error when attempting to import MDPSolver in Python. In this case, it might help to manually compile and replace the SO-file for the optimization module in the MDPSolver package. See the steps on how to solve the issue in the [documentation](https://github.com/areenberg/MDPSolver/wiki#how-to-install).

## Windows

Requires [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) (17.9) with MSVC C++ compiler and libraries installed.

After installing Visual Studio (incl. MSVC C++ compiler and libraries), install directly from PyPI with:

```
pip install mdpsolver
```

# Quick start guide

The following shows how to get quickly started with `mdpsolver`.

## Usage

Start by specifying the reward function and transition probabilities as lists. The following is an example of a simple MDP containing **three states** and **two actions in each state**.

```python
#Import packages
import mdpsolver

#Rewards (3 states x 2 actions)
#e.g. choosing second action in first state gives reward=-1
rewards = [[5,-1],
           [1,-2],
           [50,0]]

#Transition probabilities (3 from_states x 2 actions x 3 to_states)
#e.g. choosing first action in third state gives a probability of 0.6 of staying in third state
tranMatWithZeros = [[[0.9,0.1,0.0],[0.1,0.9,0.0]],
                    [[0.4,0.5,0.1],[0.3,0.5,0.2]],
                    [[0.2,0.2,0.6],[0.5,0.5,0.0]]]
```

Now, create the model object and insert the problem parameters.

```python
#Create model object
mdl = mdpsolver.model()

#Insert the problem parameters
mdl.mdp(discount=0.8,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
```

We can now optimize the policy.

```python
mdl.solve()
```

The optimized policy can be returned in a variety of ways. Here, we return the policy as a list and print directly in the terminal. 

```python
print(mdl.getPolicy())
#[1, 1, 0]
```

## Sparse transition matrix?

`mdpsolver` has three alternative formats for large and highly sparse transition probability matrices.

**(1) Elementwise representation (excluding elements containing zeros):**

```python
#[from_state,action,to_state,probability]
tranMatElementwise = [[0,0,0,0.9],
                      [0,0,1,0.1],
                      [0,1,0,0.1],
                      [0,1,1,0.9],
                      [1,0,0,0.4],
                      [1,0,1,0.5],
                      [1,0,2,0.1],
                      [1,1,0,0.3],
                      [1,1,1,0.5],
                      [1,1,2,0.2],
                      [2,0,0,0.2],
                      [2,0,1,0.2],
                      [2,0,2,0.6],
                      [2,1,0,0.5],
                      [2,1,1,0.5]]

mdl.mdp(discount=0.8,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
```

**(2) Probabilities and column (to_state) indices in separate lists:**

```python
tranMatProbs = [[[0.9,0.1],[0.1,0.9]],
                [[0.4,0.5,0.1],[0.3,0.5,0.2]],
                [[0.2,0.2,0.6],[0.5,0.5]]]

tranMatColumns = [[[0,1],[0,1]],
                [[0,1,2],[0,1,2]],
                [[0,1,2],[0,1]]]

mdl.mdp(discount=0.8,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)

```

**(3) Load the elementwise representation from a file:**

```python
mdl.mdp(discount=0.8,
        rewards=rewards,
        tranMatFromFile="transitions.csv")
```

# Documentation

The documentation can be found in the [wiki for MDPSolver](https://github.com/areenberg/MDPSolver/wiki).

# How to cite

[![DOI](https://zenodo.org/badge/294063917.svg)](https://zenodo.org/badge/latestdoi/294063917)
