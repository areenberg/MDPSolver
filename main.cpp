/*
* File:   main.cpp
* Author: jfan
*
* Created on 18. september 2020, 12:00
*/

#include <cstdlib>
#include <iostream>
#include <string>
#include "TBMModel.h"
#include "modifiedPolicyIteration.h"

using namespace std;

int main(int argc, char** argv) {
	
	//Solve Time-Based Maintenance (TBM) replacement problem
	int N = 2; //two components
	int L = 10; //maximum component age
	double discount = 0.99; //discount factor
	
	// generate model object
	Model mdl(N, L, discount);

	//Solver arguments
	double epsilon = 1e-3; //value function precision
	string algorithm = "MPI"; //VI, PI, or MPI
	string update = "SOR"; //Standard, GS (Gauss-Seidel), or SOR (Successive Over-Relaxation)
	int M = 100; //Partial evaluation iteration limit
	double SORrelaxation = 1.1; //SOR relaxation parameter
	//create solver object
	modifiedPolicyIteration mpi(mdl, epsilon, algorithm, update, M, SORrelaxation);

	mpi.solve(mdl);

	cout << endl << "Optimal policy";

	for (int sidx = 0; sidx < mdl.numberOfStates; ++sidx) {
		if (sidx % (mdl.L + 1) == 0) {
			cout << endl;
		}
		cout << mdl.policy[sidx] << " ";
	}
	cout << endl;

	return 0;
}

