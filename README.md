# `mdpsolver`: An efficient solver for Markov Decision Processes
 
`mdpsolver` is the Python package for deriving the policies of Markov Decision Processes (MDPs) with infinite-horizon and the expected total discounted reward optimality criterion. 

## Features

* Available on PyPI.
* Solver-engine developed in C++.
* Available optimization algorithms: *Value iteration*, *Policy iteration*, and *Modified policy iteration*.  
* Includes a variety of input formats available for users to choose from.

# Quick start guide

The following shows how to get quickly started with `mdpsolver`.

## Installation

Download and install `mdpsolver` directly from PyPI. 

```
pip install mdpsolver
```

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

## Large transition matrix?

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

# User manual

Further information can be found in the wiki for `mdpsolver` on Github.

# How to cite

[![DOI](https://zenodo.org/badge/294063917.svg)](https://zenodo.org/badge/latestdoi/294063917)