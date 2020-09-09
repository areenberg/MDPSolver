/*
 * Copyright 2019 Anders Reenberg Andersen and Jesper Fink Andersen
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
 * File:   modifiedPolicyIteration.cpp
 * Author: $Anders Reenberg Andersen and Jesper Fink Andersen
 * 
 * Created on 20. november 2019, 12:32
 */

#include "modifiedPolicyIterationSOR.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <math.h>
#include <fstream>

using namespace std;

modifiedPolicyIteration::modifiedPolicyIteration(Model& model, double epsilon, bool useSpan, int update, int M, double SORrelaxation):
//initialize vectors
nz(model.numberOfStates,0),
//span stopping criteria (only if using standard update)
useSpan(useSpan && update==1),
//use 1=STANDARD, 2=GS (Gauss-seidel), or 3=SOR updatas 
update(update),
SORrelaxation(SORrelaxation),
//others
epsilon(epsilon), //tolerance
M(M), //partial policy eval. iteration limit
printStuff(true),
iterLim((int) 1e5), //iteration limit
duration(0),
converged(false),
k(0),
PIconvergence(false) //experiment with terminating only when policy does not change
{ 
}

modifiedPolicyIteration::modifiedPolicyIteration(const modifiedPolicyIteration& orig) {
}

modifiedPolicyIteration::~modifiedPolicyIteration() {
}

void modifiedPolicyIteration::solve(Model& model){
	// - Standard, Gauss-Seidel, or SOR updates. (only for discounted reward criterion)
	// - VI (M=0), PI (M="infinity")
	// - All probabilities are calculated "on demand".

    //initialize value vectors and pointers
	v.assign(model.numberOfStates, 0);
	initValue(model); //step 1 in Puterman page 213. Initializes v,diffMax,diffMin, and policy
	
	vp = &v;
	if (update==1) { 
		v2 = v; //copy contents of v into v2
		vpOld = &v2;
	} else { //We only need to store one v if using GS or SOR updates
		vpOld = vp; //point to v to use gauss-seidel
	}

	//initialize tolerance depending on stopping criteria
	discount = model.discount;
	if (useSpan) {
		tolerance = epsilon * (1 - model.discount) / model.discount; //tolerance for span
	} else {
		tolerance = epsilon * (1 - model.discount) / (2 * (model.discount)); //tolerance for sup norm
	}
	
	norm = numeric_limits<double>::infinity(); //span or supremum norm

    auto t1 = chrono::high_resolution_clock::now(); //start time
    
	if (printStuff) {
		cout << "solving with ";
		if (update == 1) {
			cout << "standard ";
		} else if (update == 2) {
			cout << "Gauss-Seidel";
		} else {
			cout << "SOR";
		}
		cout << " updates and ";
		if (useSpan) {
			cout << "span" << endl;
		} else {
			cout << "supremum" << endl;
		}

	} 

	iter = 0;
	//while (norm >= tolerance && iter < iterLim) { //MAIN LOOP
	while (iter < iterLim && !PIconvergence) { //PI convergence attempt
		if (printStuff) {
			if (M > 0) {
				cout << iter << ", current v[0]: " << (*vpOld)[0] << ", norm: " << norm << ", nChanges: " << nChanges << " mn " << k << endl;
			} else if (iter % 100 == 0) {
				cout << iter << ", current v[0]: " << (*vpOld)[0] << ", norm: " << norm << endl;
			}
		}

		//PARTIAL EVALUATION
		if (update < 3) {
			partialEvaluation(model);
		} else {
			partialEvaluationSOR(model);
		}

		//POLICY IMPROVEMENT
		if (update <= 3) {
			improvePolicy(model); 
		}
		else {
			improvePolicySOR(model);
		}
		
		iter++;

	}

	//make sure v is the last updated vector if we use standard updates
	if (update==1 && vpOld != &v) { //vpOld points to last updated value vector at this point
		v = v2; //copy content of v2 into v
	}

	//if span AND discounted reward are used alter v using 6.6.12 in Puterman
	if (useSpan) {
		if (printStuff) { cout << "corrected v according to eq. (6.6.12) in Puterman" << endl; }
		for (double& val : v) {
			val += model.discount / (1 - model.discount) * diffMin;
		}
	}

    auto t2 = chrono::high_resolution_clock::now(); //stop time
	duration = (double) chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();
	
	if (printStuff) { cout << "v = " << v[0] << " " << v[1] << " " << v[2] << endl; }

	if (iter == iterLim) {
		if (printStuff) { cout << "Modified policy iteration terminated at iteration limit" << endl; }
	} else {
		converged = true;
		if (printStuff) { cout << "Modified policy iteration finished in " << iter << " iterations and " << duration << " milliseconds" << endl; }
	}
    
	checkFinalValue(model);

}


void modifiedPolicyIteration::improvePolicy(Model& model) {
	//improves the policy based on the current v

	nChanges = 0;

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();

	for (sidx = 0; sidx < model.numberOfStates; sidx++) {

		//maximize v getting the current best action
		bestVal = -numeric_limits<double>::infinity();
		for (aidx = 0; aidx < model.numberOfActions; aidx++) { //enumerates all actions
			sm = 0;
			pdidx = model.postDecisionIdx(sidx, aidx);
			model.transProb(sidx, aidx, pdidx);
			do {
				sm += model.psj * (*vpOld)[model.nextState];
				model.updateTransProbNextState(sidx, aidx, model.nextState);
			} while (model.nextState != pdidx);
			val = model.reward(sidx, aidx) + discount * sm;
			if (val > bestVal) {
				bestVal = val;
				bestAidx = aidx;
			}
		}
		//update action if necessary
		if (model.policy[sidx] != bestAidx) {
			nChanges++;
			model.policy[sidx] = bestAidx;
		}
		updateNorm();
		(*vp)[sidx] = bestVal;
	}
	swapPointers(); //for standard updates
	if (M == 1000000 && nChanges == 0) {
		PIconvergence = true;
	}
}

void modifiedPolicyIteration::partialEvaluation(Model& model){
	for (k = 0; k < M; k++){ 
		if ( norm >= tolerance ) { //We always allow early termination before M iterations
			norm = 0;
			diffMax = -numeric_limits<double>::infinity();
			diffMin = numeric_limits<double>::infinity();
			for (sidx = 0; sidx < model.numberOfStates; sidx++) {
				sm = 0;
				pdidx = model.postDecisionIdx(sidx, model.policy[sidx]);
				model.transProb(sidx, model.policy[sidx], pdidx);
				do {
					sm += model.psj * (*vpOld)[model.nextState];
					model.updateTransProbNextState(sidx, model.policy[sidx], model.nextState);
				} while (model.nextState != pdidx);
				bestVal = model.reward(sidx, model.policy[sidx]) + discount * sm;
				updateNorm();
				(*vp)[sidx] = bestVal;
			}
			swapPointers(); //for standard update	
		} else {
			break; //stop partial evaluation earlier
		}
	}
}


void modifiedPolicyIteration::improvePolicySOR(Model& model) {
	//improves the policy based on the current v

	nChanges = 0;

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();

	for (sidx = 0; sidx < model.numberOfStates; sidx++) {

		//maximize v getting the current best action
		bestVal = -numeric_limits<double>::infinity();
		for (aidx = 0; aidx < model.numberOfActions; aidx++) { //enumerates all actions
			sm = 0;
			pdidx = model.postDecisionIdx(sidx, aidx);
			model.transProb(sidx, aidx, pdidx);
			do {
				if (model.nextState != sidx) { //skip diagonal element
					sm += model.psj * (*vpOld)[model.nextState];
				}
				model.updateTransProbNextState(sidx, aidx, model.nextState);
			} while (model.nextState != pdidx);
			val = (1 - SORrelaxation) * (*vpOld)[sidx] +
				SORrelaxation / (1 - model.discount * model.transProb(sidx, aidx, sidx)) *
				(model.reward(sidx, aidx) + model.discount * sm); //SOR equation in paper
			if (val > bestVal) {
				bestVal = val;
				bestAidx = aidx;
			}
		}
		//update action if necessary
		if (model.policy[sidx] != bestAidx) {
			nChanges++;
			model.policy[sidx] = bestAidx;
		}
		updateNorm();
		(*vp)[sidx] = bestVal;
	}
	if (M == 1000000 && nChanges == 0) {
		PIconvergence = true;
	}
}

void modifiedPolicyIteration::partialEvaluationSOR(Model& model) {
	for (k = 0; k < M; k++) {
		if (norm >= tolerance) { //We always allow early termination before M iterations
			norm = 0;
			diffMax = -numeric_limits<double>::infinity();
			diffMin = numeric_limits<double>::infinity();

			for (sidx = 0; sidx < model.numberOfStates; sidx++) {
				sm = 0;
				pdidx = model.postDecisionIdx(sidx, model.policy[sidx]);
				model.transProb(sidx, model.policy[sidx], pdidx);
				do {
					if (model.nextState != sidx) { //skip diagonal element
						sm += model.psj * (*vpOld)[model.nextState];
					}
					model.updateTransProbNextState(sidx, model.policy[sidx], model.nextState);
				} while (model.nextState != pdidx);
				bestVal = (1 - SORrelaxation) * (*vpOld)[sidx] +
					SORrelaxation / (1 - model.discount * model.transProb(sidx, model.policy[sidx], sidx)) *
					(model.reward(sidx, model.policy[sidx]) + model.discount * sm); //SOR equation in paper
				updateNorm(); //this uses bestVal and not val
				(*vp)[sidx] = bestVal;
			}
		} else {
			break; //stop partial evaluation earlier
		}
	}
}

void modifiedPolicyIteration::initValue(Model& model){
    //step 1 on algorithm on page 213.
	//initializing the value vector, v, such that Bv>0

	//initialize policy as maximum reward in each state
	double maxMaxRew = -numeric_limits<double>::infinity();
	double minMaxRew = numeric_limits<double>::infinity();
	double maxRew;
	for (sidx = 0; sidx < model.numberOfStates; ++sidx) {
		
		maxRew = -numeric_limits<double>::infinity();
		
		for (aidx = 0; aidx < model.numberOfActions; ++aidx) {
			if (model.reward(sidx, aidx) > maxRew) { 
				model.policy[sidx] = aidx; //argmax_a r(s,a)
				maxRew = model.reward(sidx, aidx); //max_a r(s,a)
			}
		}

		if (maxRew < minMaxRew) {
			minMaxRew = maxRew;
		}
		if (maxRew > maxMaxRew) {
			maxMaxRew = maxRew;
		}

		v[sidx] = maxRew;
	}

	//initialize v
	for (int sidx = 0; sidx < model.numberOfStates; ++sidx) {
		v[sidx] += model.discount / (1 - model.discount) * minMaxRew;
	}

	//initialize  diffMin and diffMax
	diffMin = 0;
	diffMax = maxMaxRew - minMaxRew;
}

void modifiedPolicyIteration::swapPointers() {
	//swap pointers so that vp becomes vpOld and vice verca
	//used in policy improvement and
	//used in partial evaluation when not using Gauss-Seidel
	vpTemp = vp;
	vp = vpOld;
	vpOld = vpTemp;
}

void modifiedPolicyIteration::updateNorm() {
	//calculate difference from last iteration and update diffMax, diffMin, and supNorm
	diff0 = bestVal - (*vpOld)[sidx];
	if (diff0>diffMax) {
		diffMax = diff0;
	}
	if (diff0<diffMin) {
		diffMin = diff0;
	}
	if (useSpan) {
		norm = diffMax - diffMin;
	} else { //supremum norm
		if (fabs(diff0) > norm) {
			norm = fabs(diff0);
		}
	}
}

void modifiedPolicyIteration::checkFinalValue(Model& model) {
	//see if final value vector is within reason

	//derive minimum reward
	double minRew = 0;
	double r;

	for (int sidx = 0; sidx < model.numberOfStates; sidx++) {
		for (int aidx = 0; aidx < model.numberOfActions; aidx++) {
			r = model.reward(sidx, aidx);
			if (r < minRew) {
				minRew = r;
			}
		}
	}
	if (minRew == -numeric_limits<double>::infinity()) {
		minRew = -1e4; // some large negative value
	}
	//smallest possible value in value vector
	minRew *= 1 / (1 - model.discount);

	for (int sidx = 0; sidx < model.numberOfStates; ++sidx) {
		if (isnan(v[sidx]) || v[sidx] < minRew || v[sidx] > 0) {
			cout << "NOT CONVERGED! Final value vector is crazy at v[" << sidx << "] = " << v[sidx] << endl;
			converged = false;
			break;
		}
	}
}