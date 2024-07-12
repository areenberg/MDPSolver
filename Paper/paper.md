---
title: 'MDPSolver: An Efficient Solver for Markov Decision Processes'
tags:
  - Python
  - Markov Decision Process
  - Optimization
  - Stochasticity
  - Dynamic Programming
authors:
  - name: Anders Reenberg Andersen
    orcid: 0000-0001-7146-3086
    equal-contrib: true
    affiliation: 1
  - name: Jesper Fink Andersen
    orcid: 0000-0002-3213-5853
    equal-contrib: true
    affiliation: 1
  
affiliations:
 - name: Department of Applied Mathematics and Computer Science, Technical University of Denmark, Denmark
   index: 1
date: 12 July 2024
bibliography: paper.bib
---

# Summary

MDPSolver is a Python package for easy and fast optimization of Markov Decision Process (MDP) problems. Thus, our package is relevant to any decision-making problem where sequences of actions need to be taken. Still, the exact set of possible actions is uncertain and revealed to the decision-maker one at a time. This specific type of decision-making problem is often denoted as an MDP or a discrete stochastic dynamic programming problem. As an example, consider the sale and replenishment of goods from a wholesale inventory. Once a day, the wholesaler observes the amount of goods in the inventory and decides whether to create a replenishment order. Creating the order too early leads to an excess amount of goods, and unnecessary holding costs, whereas creating the order too late leads to shortage and lost sales. With MDPSolver, the wholesaler will be able to maximize profit by deriving optimized actions for each level of the remaining goods in the inventory.

# Statement of need

The industry currently experiences an increasing availability of data and computational resources. This development has created an opportunity for optimized decision-making based on mathematical modeling. Conversely, computational implementations can be costly, opening the door to software packages containing efficient and easily accessible tools. MDPSolver addresses this need by providing the accessibility of algorithms, optimality criteria, and other configurations. MDPSolver is highly applicable as a research tool since users can quickly employ and test various models and solution approaches. Due to the straightforward applicability of MDP models in decision-making sciences, such as Operations Research, MDPSolver has an important role in achieving any goals related to optimized decision-making under uncertainty.

Several similar software packages are available for Python [@pymdptoolbox:2015], R [@MDPtoolbox:2017], Matlab [@Matlabtoolbox:2015], and Julia [@pomdps:2017] (see the section on related software below). MDPSolver targets Python and improves on the *pymdptoolbox* package [@pymdptoolbox:2015] by providing a parallelized efficient implementation of algorithms from MDP theory [@Puterman:1994a]. Our numerical experiments show that our implementation results in a substantially shorter runtime. Moreover, MDPSolver provides the user with the option to experiment with various value update methods that are not available in the *pymdptoolbox* package. In an upcoming version, MDPSolver will also contain a selection of built-in MDP problems often considered in the scientific literature, such as multi-component replacement, job scheduling, and inventory management. The built-in problems will enable users to evaluate even larger models by computing the model parameters (e.g. transition probabilities) on demand.

In our past research, @Andersen:2022b and @Andersen:2022a used earlier versions of MDPSolver for conducting numerical experiments on multi-component replacement problems. In the future, we are planning on utilizing MDPSolver for conducting experiments in our research on the optimization of hospital patient flow.

# Features

MDPSolver enables users to derive epsilon-optimal policies for infinite horizon MDP models in Python. Users can choose to derive the policy using the value iteration, policy iteration, or modified policy iteration algorithm with the discounted or average reward optimality criterion. Furthermore, MDPSolver enables users to choose between standard, Gauss-Seidel, and Successive Over-Relaxation value updates. Choosing the standard value updates will activate parallel computing by default. In addition, to conserve computational memory, we have implemented three input options for sparse transition probability matrices, including the option to use the full matrix for teaching purposes. MDPSolver can read and write parameters and results directly from and to text files on the computer.

# Related software

The following list contains related software packages that are available for the Python, R, Matlab, and Julia languages. 

* Python: **pymdptoolbox** [@pymdptoolbox:2015]. Available on the *Python Package Index* (PyPI).
* R: **MDPtoolbox** [@MDPtoolbox:2017]. Available on the *Comprehensive R Archive Network* (CRAN).
* Matlab: **Markov Decision Processes (MDP) Toolbox** [@Matlabtoolbox:2015]. Available on the *Matlab File Exchange*. See also the built-in **createMDP** function [@createMDP].
* Julia: **POMDPs.jl** [@pomdps:2017]. Available on *Julia Packages*.

# Acknowledgements

The authors appreciate the funding and resources provided by the Technical University of Denmark which led to the completion of MDPSolver's initial version.

# References