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


ModifiedPolicyIteration::ModifiedPolicyIteration(double epsilon, string algorithm,
	string update, int parIterLim, double SORrelaxation, bool verbose, bool postProcessing, bool makeFinalCheck, bool genMDP):
	epsilon(epsilon),
	useMPI(algorithm.compare("mpi") == 0),
	usePI(algorithm.compare("pi") == 0),
	useVI(algorithm.compare("vi") == 0),
	useStd(update.compare("standard") == 0),
	useGS(update.compare("gs") == 0),
	useSOR(update.compare("sor") == 0),
	parIterLim(parIterLim), //partial evaluation iteration limit in MPI
	SORrelaxation(SORrelaxation),
	//others
	iterLim((int)1e6), //iteration limit
	PIparIterLim((int)1e6), //iteration limit for policy evaluation in PI
	initPol(false),
	initVal(false),
	genMDP(genMDP),
	printStuff(verbose), //set "true" to print algorithm progress at runtime
	postProcessing(postProcessing),
	makeFinalCheck(makeFinalCheck),
	duration(0.0),
	converged(false),
	parIter(0)
{
	//check valid string input
	assert(update.compare("standard")==0 || update.compare("gs")==0 || update.compare("sor")==0);
	assert(algorithm.compare("vi")==0 || algorithm.compare("pi")==0 || algorithm.compare("mpi")==0);
}


ModifiedPolicyIteration::ModifiedPolicyIteration(const ModifiedPolicyIteration& orig) {
}

ModifiedPolicyIteration::~ModifiedPolicyIteration() {
}


void ModifiedPolicyIteration::solve(ModelType * mdl, Policy * ply, ValueVector * vv){
	//The MDP is solved using the expected total discounted reward criterion.
	//All probabilities and rewards are calculated "on demand".

    //initialize value vectors and their pointers
    model = mdl;
    policy = ply;
	valueVector = vv;
	if (policy->policy.size()==1&&policy->policy[0]==-1){
		policy->setSize(*model->getNumberOfStates());
		initPol=true;
	}
	if (valueVector->valueVector.size()==1&&valueVector->valueVector[0]==-1){
		valueVector->setSize(*model->getNumberOfStates());
		initVal=true;
	}
	//v.assign(model.getNumberOfStates(), 0);
	initValue(); //step 1 in Puterman page 213. Initializes v,diffMax,diffMin, and policy
	vp = &valueVector->valueVector;
	if (useStd) {
		v2 = valueVector->valueVector; //copy contents of v into v2
		vpOld = &v2;
	} else { //We only need to store one v if using GS or SOR updates
		vpOld = vp; //point to v to use gauss-seidel
	}

	//initialize tolerance depending on stopping criteria
	if (useStd) {
		tolerance = epsilon * (1 - *model->getDiscount()) / *model->getDiscount(); //tolerance for span
	} else {
		tolerance = epsilon * (1 - *model->getDiscount()) / (2 * (*model->getDiscount())); //tolerance for sup norm
	}

	//change partial iteration limit if using VI or PI
	if (useVI) {
		parIterLim = 0;
	} else if (usePI) {
		parIterLim = PIparIterLim;
	}

	if (printStuff) {
		cout << "Solving with ";
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
			cout << "Span";
		} else {
			cout << "Supremum";
		}
		cout << " norm stopping criterion." << endl;
	}

	//MAIN LOOP
	
	auto t1 = chrono::high_resolution_clock::now(); //start timer

	if (!useVI){
		mainLoopModifiedPolicyIteration();
	}else{
		mainLoopValueIteration();
	}	

    auto t2 = chrono::high_resolution_clock::now(); //stop time
	duration = (double) chrono::duration_cast<chrono::nanoseconds>( t2 - t1 ).count() / 1e6;

	//POST PROCESSING

	//make sure v is the last updated vector if we use standard updates
	if (postProcessing && useStd && vpOld != &valueVector->valueVector) { //vpOld points to last updated value vector at this point
		valueVector->valueVector = v2; //copy content of v2 into v
	}

	//if using span stopping criterion alter final v using 6.6.12 in Puterman
	if (postProcessing && useStd) {
		if (printStuff) { cout << "Corrected values according to Eq. (6.6.12) in M. L. Puterman, 'Markov Decision Processes: Discrete Stochastic Dynamic Programming', Wiley." << endl; }
		for (double& val : valueVector->valueVector) {
			val += *model->getDiscount() / (1 - *model->getDiscount()) * diffMin;
		}
	}

	if (printStuff) {
		if (valueVector->valueVector.size()>3){
			cout << "v = " << valueVector->valueVector[0] << " " << valueVector->valueVector[1] << " " << valueVector->valueVector[2] << " ... " << valueVector->valueVector[valueVector->valueVector.size()-1] << endl;		
		}else{
			cout << "v = ";
			for (int sidx=0; sidx<valueVector->valueVector.size(); sidx++){
				cout << valueVector->valueVector[sidx] << " ";
			}
			cout << endl;
		}			 
	}

	if(iter == iterLim && printStuff){
		cout << "Algorithm terminated at iteration limit." << endl;
	}else if (printStuff){
		converged = true;
		cout << "Solution found in " << iter << " iterations and " << duration << " milliseconds." << endl;
	}

	if (makeFinalCheck){
		checkFinalValue();
	}

}


void ModifiedPolicyIteration::mainLoopModifiedPolicyIteration(){
	//MAIN LOOP for policy iteration and modified policy iteration

	norm = numeric_limits<double>::infinity();
	polChanges = numeric_limits<int>::infinity();
	iter = 0;
	do{
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
		if (!useSOR && genMDP) {
			partialEvaluationGenMDP();
		}else if(!useSOR && !genMDP){
			partialEvaluation();
		}else if(useSOR && genMDP){
			partialEvaluationSORGenMDP();
		}else if(useSOR && !genMDP){
			partialEvaluationSOR();
		}

		//POLICY IMPROVEMENT
		if (!useSOR && genMDP) {
			improvePolicyGenMDP();
		}else if(!useSOR && !genMDP){
			improvePolicy();
		}else if(useSOR && genMDP){
			improvePolicySORGenMDP();
		}else if(useSOR && !genMDP){
			improvePolicySOR();
		}

		iter++;

	}while( (!usePI && norm >= tolerance && iter < iterLim) || (usePI && polChanges>0) );

}

void ModifiedPolicyIteration::mainLoopValueIteration(){
	//main loop for value iteration

	norm = numeric_limits<double>::infinity();
	iter = 0;
	do{
		if (printStuff) {
			cout << iter << ", current v[0]: " << (*vpOld)[0] << ", norm: " << norm << endl;
		}

		//EVALUATION
		if (!useSOR && genMDP) {
			valueIterationEvaluationGenMDP();
		}else if(!useSOR && !genMDP){
			valueIterationEvaluation();
		}else if(useSOR && genMDP){
			valueIterationEvaluationSORGenMDP();
		}else if(useSOR && !genMDP){
			valueIterationEvaluationSOR();
		}

		iter++;
		
	}while(norm >= tolerance && iter < iterLim);

	//GET POLICY
	if (!useSOR && genMDP) {
		valueIterationPolicyGenMDP();
	}else if(!useSOR && !genMDP){
		valueIterationPolicy();
	}else if(useSOR && genMDP){
		valueIterationPolicySORGenMDP();
	}else if(useSOR && !genMDP){
		valueIterationPolicySOR();
	}

}


void ModifiedPolicyIteration::improvePolicy() {
	//improves the policy based on the current v

	polChanges = 0;
	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			sf = *model->postDecisionIdx(sidx, aidx);
			model->transProb(sidx, aidx, sf);
			do {
				valSum += *model->getPsj() * (*vpOld)[*model->getNextState()];
				model->updateNextState(sidx, aidx, *model->getNextState());
			} while (*model->getNextState() != sf);
			val = *model->reward(sidx, aidx) + *model->getDiscount() * valSum;
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		//update policy if necessary
		if (*policy->getPolicy(sidx) != aBest) {
			polChanges++;
			policy->assignPolicy(sidx,aBest);
		}
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
	swapPointers(); //for standard updates
}

void ModifiedPolicyIteration::improvePolicyGenMDP() {
	//improves the policy based on the current v

	polChanges = 0;
	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			nJumps = model->getNumberOfJumps(sidx,aidx);
			for (cidx=0; cidx<nJumps; cidx++){
				valSum += *model->transProb(sidx, aidx, cidx) * (*vpOld)[*model->getColumnIdx(sidx, aidx, cidx)];
			}
			val = *model->reward(sidx, aidx) + *model->getDiscount() * valSum;
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		//update policy if necessary
		if (*policy->getPolicy(sidx) != aBest) {
			polChanges++;
			policy->assignPolicy(sidx,aBest);
		}
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
	swapPointers(); //for standard updates
}



void ModifiedPolicyIteration::partialEvaluation(){
	
	for (parIter = 0; parIter < parIterLim; parIter++){
		if ( norm >= tolerance ) { //We allow early termination before parIterLim iterations
			norm = 0;
			diffMax = -numeric_limits<double>::infinity();
			diffMin = numeric_limits<double>::infinity();
			for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
				valSum = 0;
				sf = *model->postDecisionIdx(sidx, *policy->getPolicy(sidx));
				model->transProb(sidx, *policy->getPolicy(sidx), sf);
				do {
					valSum += *model->getPsj() * (*vpOld)[*model->getNextState()];
					model->updateNextState(sidx, *policy->getPolicy(sidx), *model->getNextState());
				} while (*model->getNextState() != sf);
				val = *model->reward(sidx, *policy->getPolicy(sidx)) + *model->getDiscount() * valSum;
				updateNorm(sidx, val);
				(*vp)[sidx] = val;
			}
			swapPointers(); //for standard update
		} else {
			break; //stop partial evaluation earlier
		}
	}
}

void ModifiedPolicyIteration::partialEvaluationGenMDP(){
	
	for (parIter = 0; parIter < parIterLim; parIter++){
		if ( norm >= tolerance ) { //We allow early termination before parIterLim iterations
			norm = 0;
			diffMax = -numeric_limits<double>::infinity();
			diffMin = numeric_limits<double>::infinity();
			for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
				valSum = 0;
				nJumps = model->getNumberOfJumps(sidx,*policy->getPolicy(sidx));
				for (cidx=0; cidx<nJumps; cidx++){
					valSum += *model->transProb(sidx, *policy->getPolicy(sidx), cidx) * (*vpOld)[*model->getColumnIdx(sidx, *policy->getPolicy(sidx), cidx)];
				}
				val = *model->reward(sidx, *policy->getPolicy(sidx)) + *model->getDiscount() * valSum;
				updateNorm(sidx, val);
				(*vp)[sidx] = val;
			}
			swapPointers(); //for standard update
		} else {
			break; //stop partial evaluation earlier
		}
	}
}

void ModifiedPolicyIteration::improvePolicySOR() {
	//improves the policy based on the current v

	polChanges = 0;
	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			sf = *model->postDecisionIdx(sidx, aidx);
			model->transProb(sidx, aidx, sf);
			do {
				if (*model->getNextState() != sidx) { //skip diagonal element
					valSum += *model->getPsj() * (*vpOld)[*model->getNextState()];
				}
				model->updateNextState(sidx, aidx, *model->getNextState());
			} while (*model->getNextState() != sf);
			val = (1 - SORrelaxation) * (*vpOld)[sidx] +
				SORrelaxation / (1 - *model->getDiscount() * *model->transProb(sidx, aidx, sidx)) *
				(*model->reward(sidx, aidx) + *model->getDiscount() * valSum); //SOR update equation
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		//update policy if necessary
		if (*policy->getPolicy(sidx) != aBest) {
			polChanges++;
			policy->assignPolicy(sidx,aBest);
		}
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
}

void ModifiedPolicyIteration::improvePolicySORGenMDP(){
	//improves the policy based on the current v

	polChanges = 0;
	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			nJumps = model->getNumberOfJumps(sidx,aidx);
			for (cidx=0; cidx<nJumps; cidx++){
				if (*model->getColumnIdx(sidx, aidx, cidx) != sidx) { //skip diagonal element
					valSum += *model->transProb(sidx, aidx, cidx) * (*vpOld)[*model->getColumnIdx(sidx, aidx, cidx)];
				}else{
					probSame = *model->transProb(sidx, aidx, cidx);
				}
			}
			val = (1 - SORrelaxation) * (*vpOld)[sidx] +
				SORrelaxation / (1 - *model->getDiscount() * probSame) *
				(*model->reward(sidx, aidx) + *model->getDiscount() * valSum); //SOR update equation
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		//update policy if necessary
		if (*policy->getPolicy(sidx) != aBest) {
			polChanges++;
			policy->assignPolicy(sidx,aBest);
		}
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
}

void ModifiedPolicyIteration::partialEvaluationSOR() {
	
	for (parIter = 0; parIter < parIterLim; parIter++) {
		if (norm >= tolerance) { //we allow early termination before parIterLim iterations
			norm = 0;
			diffMax = -numeric_limits<double>::infinity();
			diffMin = numeric_limits<double>::infinity();

			for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
				valSum = 0;
				sf = *model->postDecisionIdx(sidx, *policy->getPolicy(sidx));
				model->transProb(sidx, *policy->getPolicy(sidx), sf);
				do {
					if (*model->getNextState() != sidx) { //skip diagonal element
						valSum += *model->getPsj() * (*vpOld)[*model->getNextState()];
					}
					model->updateNextState(sidx, *policy->getPolicy(sidx), *model->getNextState());
				} while (*model->getNextState() != sf);
				val = (1 - SORrelaxation) * (*vpOld)[sidx] +
					SORrelaxation / (1 - *model->getDiscount() * *model->transProb(sidx, *policy->getPolicy(sidx), sidx)) *
					(*model->reward(sidx, *policy->getPolicy(sidx)) + *model->getDiscount() * valSum); //SOR equation in paper
				updateNorm(sidx, val);
				(*vp)[sidx] = val;
			}
		} else {
			break; //stop partial evaluation earlier
		}
	}
}

void ModifiedPolicyIteration::partialEvaluationSORGenMDP() {
	
	for (parIter = 0; parIter < parIterLim; parIter++) {
		if (norm >= tolerance) { //we allow early termination before parIterLim iterations
			norm = 0;
			diffMax = -numeric_limits<double>::infinity();
			diffMin = numeric_limits<double>::infinity();
			for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
				valSum = 0;
				nJumps = model->getNumberOfJumps(sidx,*policy->getPolicy(sidx));
				for (cidx=0; cidx<nJumps; cidx++){
					if (*model->getColumnIdx(sidx, *policy->getPolicy(sidx), cidx) != sidx) { //skip diagonal element
						valSum += *model->transProb(sidx, *policy->getPolicy(sidx), cidx) * (*vpOld)[*model->getColumnIdx(sidx, *policy->getPolicy(sidx), cidx)];
					}else{
						probSame = *model->transProb(sidx, *policy->getPolicy(sidx), cidx);
					}
				}
				val = (1 - SORrelaxation) * (*vpOld)[sidx] +
					SORrelaxation / (1 - *model->getDiscount() * probSame) *
					(*model->reward(sidx, *policy->getPolicy(sidx)) + *model->getDiscount() * valSum); //SOR update equation
				updateNorm(sidx, val);
				(*vp)[sidx] = val;
			}
		}else{
			break; //stop partial evaluation earlier
		}
	}
}

void ModifiedPolicyIteration::valueIterationEvaluation(){
	//update the value vector in the value iteration algorithm

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			sf = *model->postDecisionIdx(sidx, aidx);
			model->transProb(sidx, aidx, sf);
			do {
				valSum += *model->getPsj() * (*vpOld)[*model->getNextState()];
				model->updateNextState(sidx, aidx, *model->getNextState());
			} while (*model->getNextState() != sf);
			val = *model->reward(sidx, aidx) + *model->getDiscount() * valSum;
			if (val > valBest) {
				valBest = val;
			}
		}
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
	swapPointers(); //for standard updates

}

void ModifiedPolicyIteration::valueIterationEvaluationGenMDP(){
	//update the value vector in the value iteration algorithm

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			nJumps = model->getNumberOfJumps(sidx,aidx);
			for (cidx=0; cidx<nJumps; cidx++){
				valSum += *model->transProb(sidx, aidx, cidx) * (*vpOld)[*model->getColumnIdx(sidx, aidx, cidx)];
			}
			val = *model->reward(sidx, aidx) + *model->getDiscount() * valSum;
			if (val > valBest) {
				valBest = val;
			}
		}
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
	swapPointers(); //for standard updates

}

void ModifiedPolicyIteration::valueIterationPolicy(){
	//get the policy in the value iteration algorithm

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			sf = *model->postDecisionIdx(sidx, aidx);
			model->transProb(sidx, aidx, sf);
			do {
				valSum += *model->getPsj() * (*vpOld)[*model->getNextState()];
				model->updateNextState(sidx, aidx, *model->getNextState());
			} while (*model->getNextState() != sf);
			val = *model->reward(sidx, aidx) + *model->getDiscount() * valSum;
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		policy->assignPolicy(sidx,aBest);
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
	swapPointers(); //for standard updates

}


void ModifiedPolicyIteration::valueIterationPolicyGenMDP(){
	//get the policy in the value iteration algorithm

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			nJumps = model->getNumberOfJumps(sidx,aidx);
			for (cidx=0; cidx<nJumps; cidx++){
				valSum += *model->transProb(sidx, aidx, cidx) * (*vpOld)[*model->getColumnIdx(sidx, aidx, cidx)];
			}
			val = *model->reward(sidx, aidx) + *model->getDiscount() * valSum;
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		policy->assignPolicy(sidx,aBest);
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
	swapPointers(); //for standard updates

}


void ModifiedPolicyIteration::valueIterationEvaluationSOR(){
	//update the value vector in the value iteration algorithm

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			sf = *model->postDecisionIdx(sidx, aidx);
			model->transProb(sidx, aidx, sf);
			do {
				if (*model->getNextState() != sidx) { //skip diagonal element
					valSum += *model->getPsj() * (*vpOld)[*model->getNextState()];
				}
				model->updateNextState(sidx, aidx, *model->getNextState());
			} while (*model->getNextState() != sf);
			val = (1 - SORrelaxation) * (*vpOld)[sidx] +
				SORrelaxation / (1 - *model->getDiscount() * *model->transProb(sidx, aidx, sidx)) *
				(*model->reward(sidx, aidx) + *model->getDiscount() * valSum); //SOR update equation
			if (val > valBest) {
				valBest = val;
			}
		}
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
}

void ModifiedPolicyIteration::valueIterationEvaluationSORGenMDP(){
	//update the value vector in the value iteration algorithm

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			nJumps = model->getNumberOfJumps(sidx,aidx);
			for (cidx=0; cidx<nJumps; cidx++){
				if (*model->getColumnIdx(sidx, aidx, cidx) != sidx) { //skip diagonal element
					valSum += *model->transProb(sidx, aidx, cidx) * (*vpOld)[*model->getColumnIdx(sidx, aidx, cidx)];
				}else{
					probSame = *model->transProb(sidx, aidx, cidx);
				}
			}
			val = (1 - SORrelaxation) * (*vpOld)[sidx] +
				SORrelaxation / (1 - *model->getDiscount() * probSame) *
				(*model->reward(sidx, aidx) + *model->getDiscount() * valSum); //SOR update equation
			if (val > valBest) {
				valBest = val;
			}
		}
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
}



void ModifiedPolicyIteration::valueIterationPolicySOR(){
	//get the policy in the value iteration algorithm

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			sf = *model->postDecisionIdx(sidx, aidx);
			model->transProb(sidx, aidx, sf);
			do {
				if (*model->getNextState() != sidx) { //skip diagonal element
					valSum += *model->getPsj() * (*vpOld)[*model->getNextState()];
				}
				model->updateNextState(sidx, aidx, *model->getNextState());
			} while (*model->getNextState() != sf);
			val = (1 - SORrelaxation) * (*vpOld)[sidx] +
				SORrelaxation / (1 - *model->getDiscount() * *model->transProb(sidx, aidx, sidx)) *
				(*model->reward(sidx, aidx) + *model->getDiscount() * valSum); //SOR update equation
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		policy->assignPolicy(sidx,aBest);
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}
}


void ModifiedPolicyIteration::valueIterationPolicySORGenMDP(){
	//get the policy in the value iteration algorithm

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < *model->getNumberOfStates(); sidx++) {
		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < *model->getNumberOfActions(); aidx++) {
			valSum = 0;
			nJumps = model->getNumberOfJumps(sidx,aidx);
			for (cidx=0; cidx<nJumps; cidx++){
				if (*model->getColumnIdx(sidx, aidx, cidx) != sidx) { //skip diagonal element
					valSum += *model->transProb(sidx, aidx, cidx) * (*vpOld)[*model->getColumnIdx(sidx, aidx, cidx)];
				}else{
					probSame = *model->transProb(sidx, aidx, cidx);
				}
			}
			val = (1 - SORrelaxation) * (*vpOld)[sidx] +
				SORrelaxation / (1 - *model->getDiscount() * probSame) *
				(*model->reward(sidx, *policy->getPolicy(sidx)) + *model->getDiscount() * valSum); //SOR update equation
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		policy->assignPolicy(sidx,aBest);
		updateNorm(sidx, valBest);
		(*vp)[sidx] = valBest;
	}

}


void ModifiedPolicyIteration::initValue(){
    //step 1 in algorithm on page 213.
	//initializing the value vector, v, such that Bv>0

	//initialize policy as maximum reward in each state
	double maxMaxRew = -numeric_limits<double>::infinity();
	double minMaxRew = numeric_limits<double>::infinity();
	double maxRew;
	for (int s = 0; s < *model->getNumberOfStates(); ++s) {

		maxRew = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(s);
		for (int a = 0; a < *model->getNumberOfActions(); a++) {
			if (*model->reward(s, a) > maxRew) {
				if (initPol){
					policy->assignPolicy(s,a); //argmax_a r(s,a)
				}
				maxRew = *model->reward(s, a); //max_a r(s,a)
			}
		}

		if (maxRew < minMaxRew) {
			minMaxRew = maxRew;
		}
		if (maxRew > maxMaxRew) {
			maxMaxRew = maxRew;
		}

		if (initVal){
			valueVector->valueVector[s] = maxRew;
		}
	}

	//initialize v
	if (initVal){
		for (int s = 0; s < *model->getNumberOfStates(); ++s) {
			valueVector->valueVector[s] += *model->getDiscount() / (1 - *model->getDiscount()) * minMaxRew;
		}
	}

	//initialize  diffMin and diffMax
	diffMin = 0;
	diffMax = maxMaxRew - minMaxRew;
}


void ModifiedPolicyIteration::checkFinalValue() {
	//See if final value vector is within reason
	//NB!! this function is specific to the TBMmodel replacement problem.

	//derive minimum reward
	double minRew = 0;
	double r;
	int s,a;

	for (s = 0; s < *model->getNumberOfStates(); s++) {
		model->updateNumberOfActions(s);
		for (a = 0; a < *model->getNumberOfActions(); a++) {
			r = *model->reward(s, a);
			if (r < minRew) {
				minRew = r;
			}
		}
	}
	if (minRew == -numeric_limits<double>::infinity()) {
		minRew = -1e4; // some large negative value
	}
	//smallest possible value in value vector
	minRew *= 1 / (1 - *model->getDiscount());

	for (s = 0; s < *model->getNumberOfStates(); ++s) {
		if (isnan(valueVector->valueVector[s]) || valueVector->valueVector[s] < minRew){
			cout << "NOT CONVERGED! Final value vector is crazy at v[" << s << "] = " << valueVector->valueVector[s] << endl;
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

void ModifiedPolicyIteration::updateNorm(int &sidx, double &val) {
	//calculate difference from last iteration and update diffMax, diffMin, and supNorm
	diff = val - (*vpOld)[sidx];
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