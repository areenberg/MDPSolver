/*
 * Copyright 2019 arean.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* 
 * File:   modifiedPolicyIteration.h
 * Author: arean
 *
 * Created on 20. november 2019, 12:32
 */

#include "TBMmodel.h"
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
    int nChanges; //count changes in policy in each iteration
    
    //medthods
    void solve(Model& model);
    
private:
    
    string algorithm; //"VI", "PI", or "MPI"
    string update; //"Standard", "GS", or "SOR"

    //parameters
    double epsilon, diff0, diffMax, norm, diffMin, tolerance, val, bestVal, sm, vjidx, discount, SORrelaxation;
    int k, sidx, aidx, pdidx, iterLim, parIterLim, PIparIterLim, bestAidx;
    bool useSpan, printStuff, PIconvergence;

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
	void updateNorm(); //updates diffMax, diffMin, and span/supNorm
    void checkFinalValue(Model& model);
};

#endif /* MODIFIEDPOLICYITERATION_H */

