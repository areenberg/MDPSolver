# MDPSolver

This repository features a C++based solver for *Markov Decision Process* (MDP) optimization problems. The solver is a *Modified Policy Iteration* (MPI) algorithm, which derives an $\epsilon$-optimal policy, where \epsilon is a tolerance parameter given to the algorithm. We further provide the user with the option to choose between two different stopping criteria, three different value update methods, as well as using the solver as a $\epsilon$-optimal *Value Iteration* (VI) or *Policy Iteration* (PI) algorithm. These options, along with a description of how to use the solver, are elaborated below.

Besides the MPI algorithm, the repository also contains two MDP-model classes which have an application in the field of Predictive Maintenance. These classes account for a condition- and time-based maintenance optimization problem, respectively.

# Getting started

In order to employ the solver, one only needs the **solver** and **model** class. To import these:
``
#include "modifiedPolicyIteration.h"
#include "CBMmodel.h"

``
Here `CBMmodel.h` accounts for the class that contains the MDP-model (in this example a condition-based maintenance problem). A guide on how to write your model-class is described below.




## Switching between VI, PI and MPI

## Update method

On how to choose the value update method.

## Stopping criteria

On how to choose the stopping criteria.

# Writing your own `Model` class

# Background theory

# License

MIT License

Copyright (c) 2020 Anders Reenberg Andersen and Jesper Fink Andersen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
