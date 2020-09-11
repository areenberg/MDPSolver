
/*
* To change this license header, choose License Headers in Project Properties.
* To change this template file, choose Tools | Templates
* and open the template in the editor.
*/

/*
* File:   partialEvaluationTest.cpp
* Author: jfan
*
* Created on 21. january 2020, 16:11
*/

#include <cstdlib>
#include <iostream>
#include "TBMModel.h"
#include "modifiedPolicyIteration.h"
#include <iomanip> //set precision
#include <string>
#include <fstream> 

using namespace std;

int main(int argc, char** argv) {
	
	//Solve Time-Based Maintenance (TBM) replacement problem
	int N = 2; //two components
	int L = 10;
	double discount = 0.99; //discount factor
	
	
	
	
	// generate model
	TBMmodel mdl(N, L, discount);

	//MPI solution
	double epsilon = 1e-3; //
	bool useSpan = true;
	int update = 1; //1=Standard, 2=Gauss-Seidel, 3=Successive Over-Relaxation
	int M = 100; //Partial evaluation iteration limit
	double SORrelaxation = 1.1;
	//		input       ( model, epsilon, useSpan, update, M, SORrelaxation)
	modifiedPolicyIteration mpi(mdl, epsilon, useSpan, update, M, SORrelaxation);

	mpi.solve(mdl);
	
	cout << update << "," << useSpan << "," << M << "," << SORrelaxation << ","
		<< N << "," << mpi.iter << "," << mpi.duration << "," << mpi.v[0] << "," << mpi.converged << "\n";

	cout << "we are done." << endl;
	return 0;
}

