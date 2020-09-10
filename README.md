# MDPSolver

This repository features a C++based solver for *Markov Decision Process* (MDP) optimization problems. The solver is a *Modified Policy Iteration* (MPI) algorithm, which derives an epsilon-optimal policy that maximizes the expected total discounted reward, where epsilon is a tolerance parameter given to the algorithm. We further provide the user with the option to choose between two different stopping criteria, three different value update methods, as well as using the solver as an epsilon-optimal *Value Iteration* (VI) or *Policy Iteration* (PI) algorithm. These options, along with a description of how to use the solver, are elaborated below.

Besides the MPI algorithm, the repository also contains two MDP-model classes which have an application in the field of Predictive Maintenance. These classes account for a condition- and time-based maintenance optimization problem, respectively.

# Getting started

In order to get started, one only needs the **solver**- and **model**-class. To import these:
```
#include "modifiedPolicyIteration.h" //import the solver
#include "CBMmodel.h" //import a model

```
Here `CBMmodel.h` accounts for the class that contains the MDP-model (in this example a condition-based maintenance problem). A guide on how to write your own model-class is described later in this file.

Now define the parameters:
```
double epsilon = 1e-3; //the tolerance
double discount = 0.99; //discount
bool useSpan = true; //use span seminorm stopping
int update = 3; //use Successive Over-Relaxation (SOR) updates
double SORrelaxation = 1.1; //SOR relaxation (only relevant if update=3)
int M = 100; //stop partial evaluation after 100 iterations

//some model specific parameters
int N = 3;
int L = 5; 
```

Next, create the model object:
```
Model mdl(N, L, discount);
```
... and plug it into the solver object
```
modifiedPolicyIteration mpi(mdl, epsilon, useSpan, update, M, SORrelaxation);
```

Note the discount parameter went into the model and not the solver. An epsilon-optimal policy can now be derived by using the `solve` method:
```
mpi.solve(mdl);
```

## Get the solution


## Switching between VI, PI and MPI

## Update method

On how to choose the value update method.

## Stopping criteria

On how to choose between span seminorm and the supremum norm stopping criteria.

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
