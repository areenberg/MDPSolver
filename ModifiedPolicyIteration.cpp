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


#include "ModifiedPolicyIteration.h"
#include <iostream>
#include <chrono>
#include <string>
#include <math.h>
#include <assert.h> //to verify "algorithm" and "update" input

using namespace std;


ModifiedPolicyIteration::ModifiedPolicyIteration(ModelType& model, double epsilon, string algorithm,
	string update, int parIterLim, double SORrelaxation):
	epsilon(epsilon),
	useMPI(algorithm.compare("MPI") == 0),
	usePI(algorithm.compare("PI") == 0),
	useVI(algorithm.compare("VI") == 0),
	useStd(update.compare("Standard") == 0),
	useGS(update.compare("GS") == 0),
	useSOR(update.compare("SOR") == 0),
	parIterLim(parIterLim), //partial evaluation iteration limit in MPI
	SORrelaxation(SORrelaxation),
	//others
	discount(model.getDiscount()),
	iterLim((int)1e5), //iteration limit
	PIparIterLim(1000000), //iteration limit for policy evaluation in PI
	printStuff(true), //set "true" to print algorithm progress at runtime
	duration(0),
	converged(false),
	parIter(0)
{
	//check valid string input
	assert(update.compare("Standard")==0 || update.compare("GS")==0 || update.compare("SOR")==0);
	assert(algorithm.compare("VI")==0 || algorithm.compare("PI")==0 || algorithm.compare("MPI")==0);
}


ModifiedPolicyIteration::ModifiedPolicyIteration(const ModifiedPolicyIteration& orig) {
}

ModifiedPolicyIteration::~ModifiedPolicyIteration() {
}


void ModifiedPolicyIteration::solve(ModelType& model, Policy& policy, ValueVector& valueVector){
	//The MDP is solved using the expected total discounted reward criterion.
	//All probabilities and rewards are calculated "on demand".

        //initialize value vectors and their pointers
        policy.setSize(model.getNumberOfStates());
        valueVector.setSize(model.getNumberOfStates());
	//v.assign(model.getNumberOfStates(), 0);
	initValue(model,policy,valueVector); //step 1 in Puterman page 213. Initializes v,diffMax,diffMin, and policy
	vp = &valueVector.valueVector;
	if (useStd) {
		v2 = valueVector.valueVector; //copy contents of v into v2
		vpOld = &v2;
	} else { //We only need to store one v if using GS or SOR updates
		vpOld = vp; //point to v to use gauss-seidel
	}

	//initialize tolerance depending on stopping criteria
	if (useStd) {
		tolerance = epsilon * (1 - model.getDiscount()) / model.getDiscount(); //tolerance for span
	} else {
		tolerance = epsilon * (1 - model.getDiscount()) / (2 * (model.getDiscount())); //tolerance for sup norm
	}

	//change partial iteration limit if using VI or PI
	if (useVI) {
		parIterLim = 0;
	} else if (usePI) {
		parIterLim = PIparIterLim;
	}

	if (printStuff) {
		cout << "solving with ";
		if (useVI) {
			cout << "VI";
		} else if (usePI) {
			cout << "PI";
		} else {
			cout << "MPI";
		}
		cout << " algorithm, ";
		if (useStd) {
			cout << "Standard";
		} else if (useGS) {
			cout << "Gauss-Seidel";
		} else {
			cout << "Successive-Over Relaxation";
		}
		cout << " value function updates, and ";
		if (useStd) {
			cout << "span";
		} else {
			cout << "supremum";
		}
		cout << " norm stopping criterion." << endl;
	}

	//MAIN LOOP
	auto t1 = chrono::high_resolution_clock::now(); //start timer
	iter = 0;
	norm = numeric_limits<double>::infinity(); //to get the while-loop started
	polChanges = 1; //to get the while-loop started
	while ( (usePI && polChanges>0) || (!usePI && norm >= tolerance && iter < iterLim) ) { //MAIN LOOP
		if (printStuff) {
			if (useMPI) {
				cout << iter << ", current v[0]: " << (*vpOld)[0] << ", norm: " << norm << ", polChanges: " << polChanges << " parIter " << parIter << endl;
			} else if (usePI) {
				cout << iter << ", current v[0]: " << (*vpOld)[0] << ", polChanges: " << polChanges << " parIter " << parIter << endl;
			} else if (iter % 100 == 0) {
				cout << iter << ", current v[0]: " << (*vpOld)[0] << ", norm: " << norm << endl;
			}
		}

		//PARTIAL EVALUATION
		if (!useSOR) {
			partialEvaluation(model,policy);
		} else {
			partialEvaluationSOR(model,policy);
		}

		//POLICY IMPROVEMENT
		if (!useSOR) {
			improvePolicy(model,policy);
		}
		else {
			improvePolicySOR(model,policy);
		}

		iter++;

	}

	//make sure v is the last updated vector if we use standard updates
	if (useStd && vpOld != &valueVector.valueVector) { //vpOld points to last updated value vector at this point
		valueVector.valueVector = v2; //copy content of v2 into v
	}

	//if using span stopping criterion alter final v using 6.6.12 in Puterman
	if (useStd) {
		if (printStuff) { cout << "corrected v according to eq. (6.6.12) in Puterman" << endl; }
		for (double& val : valueVector.valueVector) {
			val += model.getDiscount() / (1 - model.getDiscount()) * diffMin;
		}
	}

    auto t2 = chrono::high_resolution_clock::now(); //stop time
	duration = (double) chrono::duration_cast<chrono::milliseconds>( t2 - t1 ).count();

	if (printStuff) { cout << "v = " << valueVector.valueVector[0] << " " << valueVector.valueVector[1] << " " << valueVector.valueVector[2] << endl; }

	if (iter == iterLim) {
		if (printStuff) { cout << "Modified policy iteration terminated at iteration limit" << endl; }
	} else {
		converged = true;
		if (printStuff) { cout << "Modified policy iteration finished in " << iter << " iterations and " << duration << " milliseconds" << endl; }
	}

	checkFinalValue(model,valueVector);

}


void ModifiedPolicyIteration::improvePolicy(ModelType& model, Policy& policy) {
	//improves the policy based on the current v

	polChanges = 0;
	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	int sf, aBest;
	double val, valBest, valSum;
	for (int s = 0; s < model.getNumberOfStates(); s++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		for (int a = 0; a < model.getNumberOfActions(); a++) {
			valSum = 0;
			sf = model.postDecisionIdx(s, a);
			model.transProb(s, a, sf);
			do {
				valSum += model.getPsj() * (*vpOld)[model.getNextState()];
				model.updateNextState(s, a, model.getNextState());
			} while (model.getNextState() != sf);
			val = model.reward(s, a) + discount * valSum;
			if (val > valBest) {
				valBest = val;
				aBest = a;
			}
		}
		//update policy if necessary
		if (*policy.getPolicy(s) != aBest) {
			polChanges++;
			policy.assignPolicy(s,aBest);
		}
		updateNorm(s, valBest);
		(*vp)[s] = valBest;
	}
	swapPointers(); //for standard updates
}


void ModifiedPolicyIteration::partialEvaluation(ModelType& model, Policy& policy){
	int sf;
	double val, valSum;
	for (parIter = 0; parIter < parIterLim; parIter++){
		if ( norm >= tolerance ) { //We allow early termination before parIterLim iterations
			norm = 0;
			diffMax = -numeric_limits<double>::infinity();
			diffMin = numeric_limits<double>::infinity();
			for (int s = 0; s < model.getNumberOfStates(); s++) {
				valSum = 0;
				sf = model.postDecisionIdx(s, *policy.getPolicy(s));
				model.transProb(s, *policy.getPolicy(s), sf);
				do {
					valSum += model.getPsj() * (*vpOld)[model.getNextState()];
					model.updateNextState(s, *policy.getPolicy(s), model.getNextState());
				} while (model.getNextState() != sf);
				val = model.reward(s, *policy.getPolicy(s)) + discount * valSum;
				updateNorm(s, val);
				(*vp)[s] = val;
			}
			swapPointers(); //for standard update
		} else {
			break; //stop partial evaluation earlier
		}
	}
}


void ModifiedPolicyIteration::improvePolicySOR(ModelType& model, Policy& policy) {
	//improves the policy based on the current v

	polChanges = 0;
	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	int sf, aBest;
	double val, valBest, valSum;
	for (int s = 0; s < model.getNumberOfStates(); s++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		for (int a = 0; a < model.getNumberOfActions(); a++) {
			valSum = 0;
			sf = model.postDecisionIdx(s, a);
			model.transProb(s, a, sf);
			do {
				if (model.getNextState() != s) { //skip diagonal element
					valSum += model.getPsj() * (*vpOld)[model.getNextState()];
				}
				model.updateNextState(s, a, model.getNextState());
			} while (model.getNextState() != sf);
			val = (1 - SORrelaxation) * (*vpOld)[s] +
				SORrelaxation / (1 - model.getDiscount() * model.transProb(s, a, s)) *
				(model.reward(s, a) + model.getDiscount() * valSum); //SOR update equation
			if (val > valBest) {
				valBest = val;
				aBest = a;
			}
		}
		//update policy if necessary
		if (*policy.getPolicy(s) != aBest) {
			polChanges++;
			policy.assignPolicy(s,aBest);
		}
		updateNorm(s, valBest);
		(*vp)[s] = valBest;
	}
}


void ModifiedPolicyIteration::partialEvaluationSOR(ModelType& model, Policy& policy) {
	int sf;
	double val, valSum;
	for (parIter = 0; parIter < parIterLim; parIter++) {
		if (norm >= tolerance) { //we allow early termination before parIterLim iterations
			norm = 0;
			diffMax = -numeric_limits<double>::infinity();
			diffMin = numeric_limits<double>::infinity();

			for (int s = 0; s < model.getNumberOfStates(); s++) {
				valSum = 0;
				sf = model.postDecisionIdx(s, *policy.getPolicy(s));
				model.transProb(s, *policy.getPolicy(s), sf);
				do {
					if (model.getNextState() != s) { //skip diagonal element
						valSum += model.getPsj() * (*vpOld)[model.getNextState()];
					}
					model.updateNextState(s, *policy.getPolicy(s), model.getNextState());
				} while (model.getNextState() != sf);
				val = (1 - SORrelaxation) * (*vpOld)[s] +
					SORrelaxation / (1 - model.getDiscount() * model.transProb(s, *policy.getPolicy(s), s)) *
					(model.reward(s, *policy.getPolicy(s)) + model.getDiscount() * valSum); //SOR equation in paper
				updateNorm(s, val);
				(*vp)[s] = val;
			}
		} else {
			break; //stop partial evaluation earlier
		}
	}
}


void ModifiedPolicyIteration::initValue(ModelType& model, Policy& policy, ValueVector& valueVector){
    //step 1 on algorithm on page 213.
	//initializing the value vector, v, such that Bv>0

	//initialize policy as maximum reward in each state
	double maxMaxRew = -numeric_limits<double>::infinity();
	double minMaxRew = numeric_limits<double>::infinity();
	double maxRew;
	for (int s = 0; s < model.getNumberOfStates(); ++s) {

		maxRew = -numeric_limits<double>::infinity();

		for (int a = 0; a < model.getNumberOfActions(); ++a) {
			if (model.reward(s, a) > maxRew) {
				policy.assignPolicy(s,a); //argmax_a r(s,a)
				maxRew = model.reward(s, a); //max_a r(s,a)
			}
		}

		if (maxRew < minMaxRew) {
			minMaxRew = maxRew;
		}
		if (maxRew > maxMaxRew) {
			maxMaxRew = maxRew;
		}

		valueVector.valueVector[s] = maxRew;
	}

	//initialize v
	for (int s = 0; s < model.getNumberOfStates(); ++s) {
		valueVector.valueVector[s] += model.getDiscount() / (1 - model.getDiscount()) * minMaxRew;
	}

	//initialize  diffMin and diffMax
	diffMin = 0;
	diffMax = maxMaxRew - minMaxRew;
}


void ModifiedPolicyIteration::checkFinalValue(ModelType& model, ValueVector& valueVector) {
	//See if final value vector is within reason
	//NB!! this function is specific to the TBMmodel replacement problem..

	//derive minimum reward
	double minRew = 0;
	double r;

	for (int s = 0; s < model.getNumberOfStates(); s++) {
		for (int a = 0; a < model.getNumberOfActions(); a++) {
			r = model.reward(s, a);
			if (r < minRew) {
				minRew = r;
			}
		}
	}
	if (minRew == -numeric_limits<double>::infinity()) {
		minRew = -1e4; // some large negative value
	}
	//smallest possible value in value vector
	minRew *= 1 / (1 - model.getDiscount());

	for (int s = 0; s < model.getNumberOfStates(); ++s) {
		if (isnan(valueVector.valueVector[s]) || valueVector.valueVector[s] < minRew || valueVector.valueVector[s] > 0) {
			cout << "NOT CONVERGED! Final value vector is crazy at v[" << s << "] = " << valueVector.valueVector[s] << endl;
			converged = false;
			break;
		}
	}
}


void ModifiedPolicyIteration::swapPointers() {
	//swap pointers so that vp becomes vpOld and vice verca
	//used in policy improvement and
	//used in partial evaluation when not using Gauss-Seidel
	vpTemp = vp;
	vp = vpOld;
	vpOld = vpTemp;
}

void ModifiedPolicyIteration::updateNorm(int s, double val) {
	//calculate difference from last iteration and update diffMax, diffMin, and supNorm
	double diff = val - (*vpOld)[s];
	if (diff>diffMax) {
		diffMax = diff;
	}
	if (diff<diffMin) {
		diffMin = diff;
	}
	if (useStd) { //span norm
		norm = diffMax - diffMin;
	} else { //supremum norm
		if (fabs(diff) > norm) {
			norm = fabs(diff);
		}
	}
}