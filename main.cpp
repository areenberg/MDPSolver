/*
* MIT License
*
* Copyright (c) 2020 Anders Reenberg Andersen and Jesper Fink Andersen
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <cstdlib>
#include <iostream>
#include <string>
#include "TBMmodel.h" //Time-based maintenance model
//#include "CBMmodel.h" //Condition-based maintenance model
#include "modifiedPolicyIteration.h"

using namespace std;

int main(int argc, char** argv) {

	//Solve Time-Based Maintenance (TBM) replacement problem
	int N = 2; //number of components
	int L = 10; //maximum component age
	double discount = 0.99; //discount factor

	// generate model object
	Model model(N, L, discount); //Time-based maintenance model
	//Model model(N, L, discount,"pCompMat.csv"); //Condition-based maintenance model

	//Solver arguments
	double epsilon = 1e-3; //epsilon-optimal policy is found
	string algorithm = "PI"; //VI, PI, or MPI
	string update = "SOR"; //Standard, GS (Gauss-Seidel), or SOR (Successive Over-Relaxation)
	int parIterLim = 100; //Partial evaluation iteration limit
	double SORrelaxation = 1.1; //SOR relaxation parameter
	//create solver object
	modifiedPolicyIteration mpi(model, epsilon, algorithm, update, parIterLim, SORrelaxation);

	//solve the MDP
	mpi.solve(model);

	//output final policy
	cout << endl << "Optimal policy";
	for (int sidx = 0; sidx < model.numberOfStates; ++sidx) {
		if (sidx % (model.L + 1) == 0) {
			cout << endl;
		}
		cout << model.policy[sidx] << " ";
	}
	cout << endl;
	
	return 0;
}
