# MDPSolver

This repository features a C++based solver for *Markov Decision Process* (MDP) optimization problems. The solver is based on a *Modified Policy Iteration* (MPI) algorithm, which derives an epsilon-optimal policy that maximizes the expected total discounted reward, where epsilon is a tolerance parameter given to the algorithm. We further provide the user with the option to choose between three different value update methods as well as switching to an epsilon-optimal *Value Iteration* (VI) or *Policy Iteration* (PI) algorithm. These options, along with a description of how to use the solver, are elaborated below.

Besides the MPI algorithm, the repository also contains two MDP-model classes which have an application in the field of Predictive Maintenance. These classes account for a condition- and time-based maintenance optimization problem, respectively. Instructions on how to write your own model-class for the solver are presented below. 

# Getting started

In order to get started, one only needs the **solver**- and **model**-class. To import these:
```
#include "modifiedPolicyIteration.h" //import the solver
#include "CBMmodel.h" //import a model

```
Here `CBMmodel.h` accounts for the class that contains the MDP-model (in this example a condition-based maintenance problem). A guide on how to write your own model-class is described later in this file.

Now define the parameters:
```
double epsilon = 1e-3; //tolerance
double discount = 0.99; //discount
string algorithm = "MPI"; //the optimization algorithm
string update = "SOR"; //the value update method
int parIterLim = 100; //number of iterations in the partial evaluation step of MPI
double SORrelaxation = 1.1; //the SOR relaxation parameter 

//some model specific parameters
int N = 3;
int L = 5; 
```

Next, create the model object:
```
Model model(N, L, discount);
```
... and plug it into the solver object
```
modifiedPolicyIteration mpi(model,epsilon,algorithm,update,parIterLim,SORrelaxation);
```

Note the discount parameter went into the model and not the solver. The `parIterLim`-parameter is only relevant if the MPI-algorithm is selected as the optimization algorithm, and `SORrelaxation` is only relevant if SOR-updates are chosen as the value update method. 

An epsilon-optimal policy can now be derived by using the `solve` method:
```
mpi.solve(mdl);
```

## Get the solution

The resulting policy is stored in the model-object in the vector `policy`. To get the action associated with state-index `sidx` run `model.policy[sidx]`. 

The following is an example on how to print the entire policy: 

```
//output final policy
	/*
	cout << endl << "Optimal policy";
	for (int sidx = 0; sidx < model.numberOfStates; ++sidx) {
		if (sidx % (model.L + 1) == 0) {
			cout << endl;
		}
		cout << model.policy[sidx] << " ";
	}
	cout << endl;
	*/
```

## Switching between VI, PI and MPI

Class `modifiedPolicyIteration` takes *three* different optimization algorithms. These are:

* Value Iteration (VI): "VI". 

* Policy Iteration (PI): "PI".
         
* Modified Policy Iteration (MPI): "MPI".
         
         
The user can easily switch between these algorithms with `string algorithm` (e.g. to choose VI `string algorithm = "VI";`). When MPI is chosen, the user is required to specify the number of *partial evaluation* iterations. This is specified with `int parIterLim` (e.g. to conduct 100 partial evaluation iterations `int parIterLim = 100;`).

Note that all three algorithms will yield an epsilon-optimal policy. Even PI will derive the value of the policy numerically based on the update-method and epsilon (tolerance) specified by the user. We have implemented the algorithms in this way to make the solver suitable for large MDP problems.

## Update method

Class `modifiedPolicyIteration` takes *three* value update methods. These are:

* Standard: "Standard".

* Gauss-Seidel: "GS".

* Successive Over-Relaxation: "SOR".

The user can switch between these methods with `string update` (e.g. to use standard value updates `string update = "Standard";`). Note that if `string update = "SOR";` the user is required to specify the relaxation (sometimes denoted omega in the literature) with `double SORrelaxation` (e.g. to set the relaxation to 1.1 `double SORrelaxation = 1.1;`). 

## Stopping criteria

The stopping criteria is automatically selected, and the solver is equipped with both a regular *supremum norm* and a *span seminorm* when convergence is evaluated. Span seminorm often terminates the algorithm earlier than the supremum norm, but only applies when the algorithm uses standard value updates. Thus, the solver will always employ the span seminorm when standard updates are selected by the user (i.e. when `string update = "Standard";`); otherwise the solver will use the supremum norm.

# Writing your own `Model` class

# Theory and references

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
