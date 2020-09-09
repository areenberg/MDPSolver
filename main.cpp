
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
#include "Model.h"
#include "modifiedPolicyIterationSOR.h"
#include <iomanip> //set precision
#include <string>
#include <fstream> 

using namespace std;

int main(int argc, char** argv) {
	
	//Compare update schemes, standard, GS, and SOR, for VI (M=0), PI (M="infinity"), and MPI (M=100)
	int inputID, N, M, update;
	bool useSpan;
	double SORrelaxation; //if update=3 
	cin >> inputID >> N >> update >> useSpan >> M >> SORrelaxation; // read from input file
	cout << "inputID " << inputID << " N " << N
		<< " update " << update << " useSpan " << useSpan << " SORrelaxation " << SORrelaxation << endl;
	
	//default parameters used
	int L = 5;
	double discount = 0.99;
	double epsilon = 1e-3;
	
	
	// write results to a file
	cout << fixed;
	cout << setprecision(6);
	string filename = "results/resultInput" + to_string(inputID) + ".txt";
	cout << "saving to " + filename << endl;
	ofstream outfile;
	outfile.open(filename);

	string inputpath = "../.././componentProbabilities/component_p_mat_N"
		+ to_string(N) + "_L" + to_string(L) + ".txt";

	// generate model
	Model mdl(N, L, discount,inputpath);

	//MPI solution
	//		input       ( model, epsilon, useSpan, update, M, SORrelaxation)
	modifiedPolicyIteration mpi(mdl, epsilon, useSpan, update, M, SORrelaxation);

	mpi.solve(mdl);
	
	//write to file
	//inputID,useGS,useAvg,useSpan,N,iterations,duration,v0,converged //resulting combined file will have these columns
	outfile << inputID << "," << update << "," << useSpan << "," << M << "," << SORrelaxation << "," 
		<< N << "," << mpi.iter << "," << mpi.duration << "," << mpi.v[0] << "," << mpi.converged << "\n";

	// stop writing to file
	outfile.close();
	cout << update << "," << useSpan << "," << M << "," << SORrelaxation << ","
		<< N << "," << mpi.iter << "," << mpi.duration << "," << mpi.v[0] << "," << mpi.converged << "\n";

	cout << "we are done." << endl;
	return 0;
}

