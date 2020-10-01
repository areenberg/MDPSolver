/*
* File:   main.cpp
* Author: jfan
*
* Created on 18. september 2020, 12:00
*/

#include <iostream>
#include <string>
//#include "TBMModel.h"
#include "CBMmodel.h"
#include "modifiedPolicyIteration.h"

using namespace std;

int main(int argc, char** argv) {
	
	/*
	//Solve Time-Based Maintenance (TBM) replacement problem
	int N = 2; //two components
	int L = 10; //maximum component age
	double discount = 0.99; //discount factor
	
	// generate model object
	Model mdl(N, L, discount);
	*/

	//Solve Condition-Based Maintenance (TBM) replacement problem
	//(CHANGE MODEL HEADER FILE IN ModifiedPolicyIteration.h TO "CBMmodel.h"
	int N = 2; //two components
	int L = 5; //component failure limit
	double discount = 0.99; //discount factor
	string importProbPath = "./CBMexampleProb_N2_L5.txt";
	// generate model object
	Model mdl(N, L, discount, importProbPath);

	//Solver arguments
	double epsilon = 1e-3; //epsilon-optimal policy is found
	string algorithm = "PI"; //VI, PI, or MPI
	string update = "SOR"; //Standard, GS (Gauss-Seidel), or SOR (Successive Over-Relaxation)
	int M = 100; //Partial evaluation iteration limit
	double SORrelaxation = 1.1; //SOR relaxation parameter
	//create solver object
	modifiedPolicyIteration mpi(mdl, epsilon, algorithm, update, M, SORrelaxation);
	
	//solve the MDP
	mpi.solve(mdl);

	//output final policy (when N=2)
	
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

