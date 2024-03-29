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

#include "TBMmodel.h" //Time-based maintenance model
//#include "CBMmodel.h" //Condition-based maintenance model
#include <vector>
#include <string>

using namespace std;

#ifndef MODIFIEDPOLICYITERATION_H
#define MODIFIEDPOLICYITERATION_H

class modifiedPolicyIteration {
public:
    //constructor and destructor
    modifiedPolicyIteration(Model& model, double eps, string algorithm = "MPI", string update = "Standard",
        int parIterLim = 100, double SORrelaxation = 1.0);
    modifiedPolicyIteration(const modifiedPolicyIteration& orig);
    virtual ~modifiedPolicyIteration();

    //value vector
    vector<double> v;

    //other parameters
    double duration;
	int iter;
	bool converged;
    int polChanges; //count changes in policy in each iteration

    //medthods
    void solve(Model& model);

private:

    //parameters
    double epsilon, diffMax, diffMin, norm, tolerance, discount, SORrelaxation;
    int iterLim, parIter, parIterLim, PIparIterLim;
    bool useMPI, usePI, useVI, useStd, useGS, useSOR, printStuff;

	//Pointers so we don't have to copy full vectors (for standard value function updates)
	vector<double> v2; //second value vector required when using standard updates
	vector<double> *vp; //pointer to last updated v
	vector<double> *vpOld; //pointer to old v
	vector<double> *vpTemp; //temporary pointer used when swapping vp and vpOld

    //methods
	void improvePolicy(Model& model);
	void partialEvaluation(Model& model);
    void improvePolicySOR(Model& model);
    void partialEvaluationSOR(Model& model);
	void initValue(Model& model); //initializes policy, v, and span
	void swapPointers(); //swaps vp and vpOld.
	void updateNorm(int s, double valBest); //updates diffMax, diffMin, and span/supNorm
    void checkFinalValue(Model& model);
};

#endif /* MODIFIEDPOLICYITERATION_H */
