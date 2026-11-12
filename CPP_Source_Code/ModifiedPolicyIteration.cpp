/*
* MIT License
*
* Copyright (c) 2026 Anders Reenberg Andersen and Jesper Fink Andersen
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
//only the #pragma omp lines use OpenMP, so a compiler without it (e.g. Apple's clang
//without libomp) builds a serial module; CMakeLists.txt treats OpenMP as optional
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <limits>
#include <string>
#include <math.h>
#include <assert.h> //to verify "algorithm" and "update" input

using namespace std;


ModifiedPolicyIteration::ModifiedPolicyIteration(double epsilon, string algorithm, string update, string criterion,
 int parIterLim, double SORrelaxation, bool verbose, bool postProcessing, bool makeFinalCheck, bool parallel):
	epsilon(epsilon),
	useMPI(algorithm.compare("mpi") == 0),
	usePI(algorithm.compare("pi") == 0),
	useVI(algorithm.compare("vi") == 0),
	useStd(update.compare("standard") == 0),
	useDis(criterion.compare("discounted") == 0),
	useAvg(criterion.compare("average") == 0),
	useGS(update.compare("gs") == 0),
	useSOR(update.compare("sor") == 0),
	parIterLim(parIterLim), //partial evaluation iteration limit in MPI
	SORrelaxation(SORrelaxation),
	//others
	iterLim((int)1e6), //iteration limit
	PIparIterLim((int)1e6), //iteration limit for policy evaluation in PI
	initPol(false),
	initVal(false),
	norm(numeric_limits<double>::infinity()),
	parallel(parallel),
	tables(nullptr),
	printStuff(verbose), //set "true" to print algorithm progress at runtime
	postProcessing(postProcessing),
	makeFinalCheck(makeFinalCheck),
	duration(0.0),
	iter(0),
	converged(false),
	polChanges(0),
	iterationLimitReached(false),
	evaluationLimitReached(false),
	valuesOutOfRange(false),
	diverged(false),
	outOfRangeState(-1),
	outOfRangeValue(0.0),
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
	//The MDP is solved using the selected optimality criterion.
	
    //initialize value vectors and their pointers
    model = mdl;
    tables = model->tables(); //nullptr for a model that computes its transitions on the fly
    policy = ply;
	valueVector = vv;
	if (!policy->isSet){
		policy->setSize(model->getNumberOfStates());
		initPol=true;
	}
	if (!valueVector->isSet){
		valueVector->setSize(model->getNumberOfStates());
		initVal=true;
	}
	
	initValue(); //step 1 in Puterman page 213. Initializes v,diffMax,diffMin, and policy
	vp = &valueVector->valueVector;
	if (useStd) {
		v2 = valueVector->valueVector; //copy contents of v into v2
		vpOld = &v2;
	} else { //We only need to store one v if using GS or SOR updates
		vpOld = vp; //point to v to use gauss-seidel
	}

	//initialize tolerance depending on stopping criteria and update method
	if (useAvg){
		//average reward criterion
		tolerance = epsilon;
	}else if(useDis){
		//discounted reward criterion
		if (useStd){
			tolerance = epsilon * (1 - model->getDiscount()) / model->getDiscount(); //tolerance for span
		}else{
			tolerance = epsilon * (1 - model->getDiscount()) / (2 * (model->getDiscount())); //tolerance for sup norm
		}
	}

	//change partial iteration limit if using VI or PI
	if (useVI) {
		parIterLim = 0;
	} else if (usePI) {
		parIterLim = PIparIterLim;
	}

	//force discount=1 if average reward criterion
	if (useAvg){
		discount=1.0;	
	}else if (useDis){
		discount = model->getDiscount();
	}
	
	//force SOR relaxation = 1 if selected GS
	if (useGS){
		SORrelaxation=1.0;
	}

	if (printStuff) {
		cout << "Solving with ";
		if (useAvg){
			cout << "average";	
		}else if (useDis){
			cout << "discounted";
		}
		cout << " reward optimality criterion, ";
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

	nStates = model->getNumberOfStates();
	iter=0;

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
	if (postProcessing && useStd && useDis) {
		if (printStuff) { cout << "Corrected values according to Eq. (6.6.12) in M. L. Puterman, 'Markov Decision Processes: Discrete Stochastic Dynamic Programming', Wiley." << endl; }
		for (double& val : valueVector->valueVector) {
			val += model->getDiscount() / (1 - model->getDiscount()) * diffMin;
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

	iterationLimitReached = (iter >= iterLim);
	//infinite or NaN values give a norm that is not finite, and NaN ends the loops
	//as if the tolerance were reached (one scalar test, not a check of the model)
	diverged = !isfinite(norm);
	if (iterationLimitReached && printStuff){
		cout << "Algorithm terminated at iteration limit." << endl;
	}else if (evaluationLimitReached && printStuff){
		cout << "Policy evaluation terminated at its iteration limit." << endl;
	}else if (printStuff){
		cout << "Solution found in " << iter << " iterations and " << duration << " milliseconds." << endl;
	}

	if (makeFinalCheck){
		checkFinalValue();
	}
	converged = !iterationLimitReached && !evaluationLimitReached && !valuesOutOfRange && !diverged;

	//the policy and the value vector now hold a solution, from which a later run starts
	policy->isSet = true;
	valueVector->isSet = true;

}

void ModifiedPolicyIteration::print(){
	if (printStuff) {
			if (useMPI) {
				cout << iter << ", current v[0]: " << (*vpOld)[0] << ", norm: " << norm << ", polChanges: " << polChanges << " parIter " << parIter << endl;
			} else if (usePI) {
				cout << iter << ", current v[0]: " << (*vpOld)[0] << ", polChanges: " << polChanges << " parIter " << parIter << endl;
			} else if (iter % 50 == 0) {
				cout << iter << ", current v[0]: " << (*vpOld)[0] << ", norm: " << norm << endl;
			}
		}
}

void ModifiedPolicyIteration::mainLoopModifiedPolicyIteration(){
	//MAIN LOOP for policy iteration and modified policy iteration
	//(models with CSR tables run the loops that read them; the others, which compute
	//their transitions on the fly, run the loops that call the model's methods)

	if (tables != nullptr){
		if (useStd && parallel){
			parModifiedPolicyIterationCSR();
		}else if (useStd){
			modifiedPolicyIterationCSR();
		}else{ //Gauss-Seidel or SOR
			modifiedPolicyIterationSORCSR();
		}
	}else{
		if (useStd){
			modifiedPolicyIteration();
		}else{ //Gauss-Seidel or SOR
			modifiedPolicyIterationSOR();
		}
	}

}

void ModifiedPolicyIteration::mainLoopValueIteration(){
	//main loop for value iteration
	//(parallel computing is only available with standard updates,
	//so Gauss-Seidel always uses the serial in-place routine)

	if (tables != nullptr){
		if (useSOR){
			valueIterationSORCSR();
		}else if (useStd && parallel){
			parValueIterationCSR();
		}else{ //standard updates, or Gauss-Seidel in place
			valueIterationCSR();
		}
	}else{
		if (useSOR){
			valueIterationSOR();
		}else{ //standard updates, or Gauss-Seidel in place
			valueIteration();
		}
	}
}

void ModifiedPolicyIteration::valueIteration(){
	//value iteration for the built-in
	//MDP models

	do{
		norm = 0;
		diffMax = -numeric_limits<double>::infinity();
		diffMin = numeric_limits<double>::infinity();
		for (sidx = 0; sidx < nStates; sidx++) {
			//find the best action
			valBest = -numeric_limits<double>::infinity();
			model->updateNumberOfActions(sidx);
			for (aidx = 0; aidx < model->getNumberOfActions(); aidx++) {
				valSum = 0;
				sf = model->postDecisionIdx(sidx, aidx);
				model->transProb(sidx, aidx, sf);
				do{
					valSum += model->getPsj() * (*vpOld)[*model->getNextState()];
					model->updateNextState(sidx, aidx, *model->getNextState());
				}while (*model->getNextState() != sf);
				val = model->reward(sidx, aidx) + discount * valSum;
				if (val > valBest) {
					valBest = val;
				}
			}
			updateNorm(valBest);
			(*vp)[sidx] = valBest;
		}
		swapPointers(); //for standard updates

		iter++;
		print();
	}while(norm >= tolerance && iter < iterLim);

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < nStates; sidx++) {

		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < model->getNumberOfActions(); aidx++) {
			valSum = 0;
			sf = model->postDecisionIdx(sidx, aidx);
			model->transProb(sidx, aidx, sf);
			do {
				valSum += model->getPsj() * (*vpOld)[*model->getNextState()];
				model->updateNextState(sidx, aidx, *model->getNextState());
			} while (*model->getNextState() != sf);
			val = model->reward(sidx, aidx) + discount * valSum;
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		policy->assignPolicy(sidx,aBest);
		updateNorm(valBest);
		(*vp)[sidx] = valBest;
	}
	swapPointers(); //for standard updates
}


void ModifiedPolicyIteration::valueIterationSOR(){

	do{
		norm = 0;
		diffMax = -numeric_limits<double>::infinity();
		diffMin = numeric_limits<double>::infinity();
		for (sidx = 0; sidx < nStates; sidx++) {
			//find the best action
			valBest = -numeric_limits<double>::infinity();
			model->updateNumberOfActions(sidx);
			for (aidx = 0; aidx < model->getNumberOfActions(); aidx++) {
				valSum = 0;
				probSame = 0;
				sf = model->postDecisionIdx(sidx, aidx);
				model->transProb(sidx, aidx, sf);
				do{
					if (*model->getNextState() != sidx) { //skip diagonal element
						valSum += model->getPsj() * (*vpOld)[*model->getNextState()];
					}else{
						probSame=model->getPsj();
					}
					model->updateNextState(sidx, aidx, *model->getNextState());
				}while(*model->getNextState() != sf);
				val = (1 - SORrelaxation) * (*vpOld)[sidx] +
					SORrelaxation / (1 - discount * probSame) *
					(model->reward(sidx, aidx) + discount * valSum); //SOR update equation
				if (val > valBest) {
					valBest = val;
				}	
			}
			updateNorm(valBest);
			(*vp)[sidx] = valBest;
		}
		iter++;
		print();
	}while(norm >= tolerance && iter < iterLim);

	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	for (sidx = 0; sidx < nStates; sidx++) {
		//find the best action
		valBest = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < model->getNumberOfActions(); aidx++) {
			valSum = 0;
			probSame=0;
			sf = model->postDecisionIdx(sidx, aidx);
			model->transProb(sidx, aidx, sf);
			do {
				if (*model->getNextState() != sidx) { //skip diagonal element
					valSum += model->getPsj() * (*vpOld)[*model->getNextState()];
				}else{
					probSame = model->getPsj();
				}
				model->updateNextState(sidx, aidx, *model->getNextState());
			} while (*model->getNextState() != sf);
			val = (1 - SORrelaxation) * (*vpOld)[sidx] +
				SORrelaxation / (1 - discount * probSame) *
				(model->reward(sidx, aidx) + discount * valSum); //SOR update equation
			if (val > valBest) {
				valBest = val;
				aBest = aidx;
			}
		}
		policy->assignPolicy(sidx,aBest);
		updateNorm(valBest);
		(*vp)[sidx] = valBest;
	}
}


void ModifiedPolicyIteration::modifiedPolicyIteration(){
	//employ modified or common policy iteration on
	//a built-in MDP model

	do{
		for (parIter=0; parIter<parIterLim; parIter++){
			if (norm>=tolerance){ //We allow early termination before parIterLim iterations
				norm = 0;
				diffMax = -numeric_limits<double>::infinity();
				diffMin = numeric_limits<double>::infinity();
				for (sidx = 0; sidx < nStates; sidx++) {
					valSum = 0;
					sf = model->postDecisionIdx(sidx, *policy->getPolicy(sidx));
					model->transProb(sidx, *policy->getPolicy(sidx), sf);
					do {
						valSum += model->getPsj() * (*vpOld)[*model->getNextState()];
						model->updateNextState(sidx, *policy->getPolicy(sidx), *model->getNextState());
					} while (*model->getNextState() != sf);
					val = model->reward(sidx, *policy->getPolicy(sidx)) + discount * valSum;
					updateNorm(val);
					(*vp)[sidx] = val;
				}
				swapPointers(); //for standard update
			}else{
				break; //stop partial evaluation earlier
			}
		}

		if (usePI && parIter == parIterLim) { evaluationLimitReached = true; }
		polChanges = 0;
		norm = 0;
		diffMax = -numeric_limits<double>::infinity();
		diffMin = numeric_limits<double>::infinity();
		for (sidx = 0; sidx < nStates; sidx++) {

			//find the best action
			valBest = -numeric_limits<double>::infinity();
			model->updateNumberOfActions(sidx);
			for (aidx = 0; aidx < model->getNumberOfActions(); aidx++) {
				valSum = 0;
				sf = model->postDecisionIdx(sidx, aidx);
				model->transProb(sidx, aidx, sf);
				do{
					valSum += model->getPsj() * (*vpOld)[*model->getNextState()];
					model->updateNextState(sidx, aidx, *model->getNextState());
				}while(*model->getNextState() != sf);
				val = model->reward(sidx, aidx) + discount * valSum;
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
			updateNorm(valBest);
			(*vp)[sidx] = valBest;
		}
		swapPointers(); //for standard updates

		iter++;
		print();
	}while( (!usePI && norm >= tolerance && iter < iterLim) || (usePI && polChanges>0) );
}


void ModifiedPolicyIteration::modifiedPolicyIterationSOR(){
	//employ modified or common policy iteration on
	//a built-in MDP model using GS or SOR updates

	do{
		for (parIter = 0; parIter < parIterLim; parIter++) {
			if (norm >= tolerance) { //we allow early termination before parIterLim iterations
				norm = 0;
				diffMax = -numeric_limits<double>::infinity();
				diffMin = numeric_limits<double>::infinity();

				for (sidx = 0; sidx < nStates; sidx++) {
					valSum = 0;
					probSame = 0;
					sf = model->postDecisionIdx(sidx, *policy->getPolicy(sidx));
					model->transProb(sidx, *policy->getPolicy(sidx), sf);
					do {
						if (*model->getNextState() != sidx) { //skip diagonal element
							valSum += model->getPsj() * (*vpOld)[*model->getNextState()];
						}else{
							probSame = model->getPsj();
						}
						model->updateNextState(sidx, *policy->getPolicy(sidx), *model->getNextState());
					} while (*model->getNextState() != sf);
					val = (1 - SORrelaxation) * (*vpOld)[sidx] +
						SORrelaxation / (1 - discount * probSame) *
						(model->reward(sidx, *policy->getPolicy(sidx)) + discount * valSum); //SOR equation in paper
					updateNorm(val);
					(*vp)[sidx] = val;
				}
			} else {
				break; //stop partial evaluation earlier
			}
		}

		if (usePI && parIter == parIterLim) { evaluationLimitReached = true; }
		polChanges = 0;
		norm = 0;
		diffMax = -numeric_limits<double>::infinity();
		diffMin = numeric_limits<double>::infinity();
		for (sidx = 0; sidx < nStates; sidx++) {

			//find the best action
			valBest = -numeric_limits<double>::infinity();
			model->updateNumberOfActions(sidx);
			for (aidx = 0; aidx < model->getNumberOfActions(); aidx++) {
				valSum = 0;
				probSame = 0;
				sf = model->postDecisionIdx(sidx, aidx);
				model->transProb(sidx, aidx, sf);
				do {
					if (*model->getNextState() != sidx) { //skip diagonal element
						valSum += model->getPsj() * (*vpOld)[*model->getNextState()];
					}else{
						probSame = model->getPsj();
					}
					model->updateNextState(sidx, aidx, *model->getNextState());
				} while (*model->getNextState() != sf);
				val = (1 - SORrelaxation) * (*vpOld)[sidx] +
					SORrelaxation / (1 - discount * probSame) *
					(model->reward(sidx, aidx) + discount * valSum); //SOR update equation
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
			updateNorm(valBest);
			(*vp)[sidx] = valBest;
		}


		iter++;
		print();
	}while( (!usePI && norm >= tolerance && iter < iterLim) || (usePI && polChanges>0) );
}

//---------------------------------------
//  LOOPS FOR MODELS WITH CSR TABLES
//---------------------------------------
//These loops read the model's CSR tables (ModelType.h) directly, rather than calling the
//model's methods for every transition probability. They do the same arithmetic in the same
//order as the loops through the methods did, so that the values, policies and iteration
//counts come out the same. The parallel ones compute the norm in the same sweep, with
//OpenMP reductions, rather than in a second parallel region.

namespace {

//the value of state-action pair i under the values v: r(i) + discount * sum_k p(k) v(j(k))
inline double pairValue(const CSRTables& t, int64_t i, const double* v, double discount){
	double valSum = 0;
	for (int64_t k = t.pairStart[i]; k < t.pairStart[i+1]; k++){
		valSum += t.prob[k] * v[t.column[k]];
	}
	return t.reward[i] + discount * valSum;
}

//the value of action a in state s; an action the state does not have (possible only in a
//malformed model, e.g. a state without actions) has the value -infinity, as an empty
//maximum, rather than reading outside the tables
inline double actionValue(const CSRTables& t, int s, int a, const double* v, double discount){
	int64_t i = t.stateStart[s] + a;
	if (a < 0 || i >= t.stateStart[s+1]){
		return -numeric_limits<double>::infinity();
	}
	return pairValue(t, i, v, discount);
}

//the SOR value of state-action pair i of state s under the values v, which leaves out the
//probability p(s|s) of staying in s and corrects for it:
//(1-w) v(s) + w/(1 - discount p(s|s)) * (r(i) + discount * sum_{j != s} p(j) v(j));
//w = 1 gives the Gauss-Seidel update. Entries for the same next state add up, as in the
//standard updates, also on the diagonal (until 2026-11-13 the last diagonal entry replaced
//the others, which lost their probability; that mattered only for repeated entries, e.g.
//arrays padded with zeros whose padding next state is s itself)
inline double pairValueSOR(const CSRTables& t, int64_t i, int s, const double* v, double discount, double w){
	double valSum = 0;
	double probSame = 0;
	for (int64_t k = t.pairStart[i]; k < t.pairStart[i+1]; k++){
		int j = t.column[k];
		if (j != s) { //skip diagonal element
			valSum += t.prob[k] * v[j];
		}else{
			probSame += t.prob[k];
		}
	}
	return (1 - w) * v[s] + w / (1 - discount * probSame) * (t.reward[i] + discount * valSum);
}

//as actionValue(), with the SOR value
inline double actionValueSOR(const CSRTables& t, int s, int a, const double* v, double discount, double w){
	int64_t i = t.stateStart[s] + a;
	if (a < 0 || i >= t.stateStart[s+1]){
		return -numeric_limits<double>::infinity();
	}
	return pairValueSOR(t, i, s, v, discount, w);
}

}

void ModifiedPolicyIteration::valueIterationCSR(){
	//value iteration with standard updates, or with Gauss-Seidel updates, which are the
	//same loop run in place (then vp and vpOld point to the same vector)
	const CSRTables& t = *tables;

	do{
		norm = 0;
		diffMax = -numeric_limits<double>::infinity();
		diffMin = numeric_limits<double>::infinity();
		const double* vOld = vpOld->data();
		double* vNew = vp->data();
		for (int s = 0; s < nStates; s++) {
			double best = -numeric_limits<double>::infinity();
			for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
				double val = pairValue(t, i, vOld, discount);
				if (val > best) {
					best = val;
				}
			}
			updateNorm(best, vOld[s]);
			vNew[s] = best;
		}
		swapPointers();
		iter++;
		print();
	}while(norm >= tolerance && iter < iterLim);

	//get actions
	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	const double* vOld = vpOld->data();
	double* vNew = vp->data();
	int* pol = policy->policy.data();
	for (int s = 0; s < nStates; s++) {
		double best = -numeric_limits<double>::infinity();
		int aBest = 0;
		for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
			double val = pairValue(t, i, vOld, discount);
			if (val > best) {
				best = val;
				aBest = (int)(i - t.stateStart[s]);
			}
		}
		pol[s] = aBest;
		updateNorm(best, vOld[s]);
		vNew[s] = best;
	}
	swapPointers();
}

void ModifiedPolicyIteration::parValueIterationCSR(){
	//value iteration with standard updates, each sweep in parallel
	const CSRTables& t = *tables;
	const int n = nStates;
	const double disc = discount;

	do{
		const double* vOld = vpOld->data();
		double* vNew = vp->data();
		double dMax = -numeric_limits<double>::infinity();
		double dMin = numeric_limits<double>::infinity();
		#pragma omp parallel for reduction(max:dMax) reduction(min:dMin)
		for (int s = 0; s < n; s++) {
			double best = -numeric_limits<double>::infinity();
			for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
				double val = pairValue(t, i, vOld, disc);
				if (val > best) {
					best = val;
				}
			}
			double d = best - vOld[s];
			if (d > dMax) {
				dMax = d;
			}
			if (d < dMin) {
				dMin = d;
			}
			vNew[s] = best;
		}
		setNorm(dMax, dMin);
		swapPointers();
		iter++;
		print();
	}while(norm >= tolerance && iter < iterLim);

	//get actions
	const double* vOld = vpOld->data();
	double* vNew = vp->data();
	int* pol = policy->policy.data();
	double dMax = -numeric_limits<double>::infinity();
	double dMin = numeric_limits<double>::infinity();
	#pragma omp parallel for reduction(max:dMax) reduction(min:dMin)
	for (int s = 0; s < n; s++) {
		double best = -numeric_limits<double>::infinity();
		int aBest = 0;
		for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
			double val = pairValue(t, i, vOld, disc);
			if (val > best) {
				best = val;
				aBest = (int)(i - t.stateStart[s]);
			}
		}
		pol[s] = aBest;
		double d = best - vOld[s];
		if (d > dMax) {
			dMax = d;
		}
		if (d < dMin) {
			dMin = d;
		}
		vNew[s] = best;
	}
	setNorm(dMax, dMin);
	swapPointers();
}

void ModifiedPolicyIteration::modifiedPolicyIterationCSR(){
	//modified (and common) policy iteration with standard updates
	const CSRTables& t = *tables;
	int* pol = policy->policy.data();

	do{
		for (parIter = 0; parIter < parIterLim; parIter++){
			if ( norm >= tolerance ) { //We allow early termination before parIterLim iterations
				norm = 0;
				diffMax = -numeric_limits<double>::infinity();
				diffMin = numeric_limits<double>::infinity();
				const double* vOld = vpOld->data();
				double* vNew = vp->data();
				for (int s = 0; s < nStates; s++) {
					double val = actionValue(t, s, pol[s], vOld, discount);
					updateNorm(val, vOld[s]);
					vNew[s] = val;
				}
				swapPointers(); //for standard update
			} else {
				break; //stop partial evaluation earlier
			}
		}

		if (usePI && parIter == parIterLim) { evaluationLimitReached = true; }
		polChanges = 0;
		norm = 0;
		diffMax = -numeric_limits<double>::infinity();
		diffMin = numeric_limits<double>::infinity();
		const double* vOld = vpOld->data();
		double* vNew = vp->data();
		for (int s = 0; s < nStates; s++) {
			//find the best action
			double best = -numeric_limits<double>::infinity();
			int aBest = 0;
			for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
				double val = pairValue(t, i, vOld, discount);
				if (val > best) {
					best = val;
					aBest = (int)(i - t.stateStart[s]);
				}
			}
			//update policy if necessary
			if (pol[s] != aBest) {
				polChanges++;
				pol[s] = aBest;
			}
			updateNorm(best, vOld[s]);
			vNew[s] = best;
		}
		swapPointers(); //for standard updates
		iter++;
		print();
	}while( (!usePI && norm >= tolerance && iter < iterLim) || (usePI && polChanges>0) );
}

void ModifiedPolicyIteration::parModifiedPolicyIterationCSR(){
	//modified (and common) policy iteration with standard updates, each sweep in parallel
	const CSRTables& t = *tables;
	const int n = nStates;
	const double disc = discount;
	int* pol = policy->policy.data();

	do{
		for (parIter = 0; parIter < parIterLim; parIter++){
			if (norm >= tolerance) { //We allow early termination before parIterLim iterations
				const double* vOld = vpOld->data();
				double* vNew = vp->data();
				double dMax = -numeric_limits<double>::infinity();
				double dMin = numeric_limits<double>::infinity();
				#pragma omp parallel for reduction(max:dMax) reduction(min:dMin)
				for (int s = 0; s < n; s++) {
					double val = actionValue(t, s, pol[s], vOld, disc);
					double d = val - vOld[s];
					if (d > dMax) {
						dMax = d;
					}
					if (d < dMin) {
						dMin = d;
					}
					vNew[s] = val;
				}
				setNorm(dMax, dMin);
				swapPointers(); //for standard update
			} else {
				break; //stop partial evaluation earlier
			}
		}

		if (usePI && parIter == parIterLim) { evaluationLimitReached = true; }
		int changes = 0;
		const double* vOld = vpOld->data();
		double* vNew = vp->data();
		double dMax = -numeric_limits<double>::infinity();
		double dMin = numeric_limits<double>::infinity();
		#pragma omp parallel for reduction(+:changes) reduction(max:dMax) reduction(min:dMin)
		for (int s = 0; s < n; s++) {
			//find the best action
			double best = -numeric_limits<double>::infinity();
			int aBest = 0;
			for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
				double val = pairValue(t, i, vOld, disc);
				if (val > best) {
					best = val;
					aBest = (int)(i - t.stateStart[s]);
				}
			}
			//update policy if necessary
			if (pol[s] != aBest) {
				changes++;
				pol[s] = aBest;
			}
			double d = best - vOld[s];
			if (d > dMax) {
				dMax = d;
			}
			if (d < dMin) {
				dMin = d;
			}
			vNew[s] = best;
		}
		polChanges = changes;
		setNorm(dMax, dMin);
		swapPointers(); //for standard updates
		iter++;
		print();
	}while( (!usePI && norm >= tolerance && iter < iterLim) || (usePI && polChanges>0) );
}

void ModifiedPolicyIteration::valueIterationSORCSR(){
	//value iteration with SOR updates, in place
	//(Gauss-Seidel updates for value iteration run in valueIterationCSR())
	const CSRTables& t = *tables;
	double* v = vp->data(); //vp and vpOld point to the same vector

	do{
		norm = 0;
		diffMax = -numeric_limits<double>::infinity();
		diffMin = numeric_limits<double>::infinity();
		for (int s = 0; s < nStates; s++) {
			double best = -numeric_limits<double>::infinity();
			for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
				double val = pairValueSOR(t, i, s, v, discount, SORrelaxation);
				if (val > best) {
					best = val;
				}
			}
			updateNorm(best, v[s]);
			v[s] = best;
		}
		iter++;
		print();
	}while(norm >= tolerance && iter < iterLim);

	//get actions
	norm = 0;
	diffMax = -numeric_limits<double>::infinity();
	diffMin = numeric_limits<double>::infinity();
	int* pol = policy->policy.data();
	for (int s = 0; s < nStates; s++) {
		double best = -numeric_limits<double>::infinity();
		int aBest = 0;
		for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
			double val = pairValueSOR(t, i, s, v, discount, SORrelaxation);
			if (val > best) {
				best = val;
				aBest = (int)(i - t.stateStart[s]);
			}
		}
		pol[s] = aBest;
		updateNorm(best, v[s]);
		v[s] = best;
	}
}

void ModifiedPolicyIteration::modifiedPolicyIterationSORCSR(){
	//modified (and common) policy iteration with Gauss-Seidel or SOR updates, in place
	const CSRTables& t = *tables;
	double* v = vp->data(); //vp and vpOld point to the same vector
	int* pol = policy->policy.data();

	do{
		for (parIter = 0; parIter < parIterLim; parIter++) {
			if (norm >= tolerance) { //we allow early termination before parIterLim iterations
				norm = 0;
				diffMax = -numeric_limits<double>::infinity();
				diffMin = numeric_limits<double>::infinity();
				for (int s = 0; s < nStates; s++) {
					double val = actionValueSOR(t, s, pol[s], v, discount, SORrelaxation);
					updateNorm(val, v[s]);
					v[s] = val;
				}
			}else{
				break; //stop partial evaluation earlier
			}
		}

		if (usePI && parIter == parIterLim) { evaluationLimitReached = true; }
		polChanges = 0;
		norm = 0;
		diffMax = -numeric_limits<double>::infinity();
		diffMin = numeric_limits<double>::infinity();
		for (int s = 0; s < nStates; s++) {
			//find the best action
			double best = -numeric_limits<double>::infinity();
			int aBest = 0;
			for (int64_t i = t.stateStart[s]; i < t.stateStart[s+1]; i++) {
				double val = pairValueSOR(t, i, s, v, discount, SORrelaxation);
				if (val > best) {
					best = val;
					aBest = (int)(i - t.stateStart[s]);
				}
			}
			//update policy if necessary
			if (pol[s] != aBest) {
				polChanges++;
				pol[s] = aBest;
			}
			updateNorm(best, v[s]);
			v[s] = best;
		}

		iter++;
		print();
	}while( (!usePI && norm >= tolerance && iter < iterLim) || (usePI && polChanges>0) );
}

void ModifiedPolicyIteration::initValue(){
    //step 1 in algorithm on page 213.
	//initializing the value vector, v, such that Bv>0

	//initialize policy as maximum reward in each state
	double maxMaxRew = -numeric_limits<double>::infinity();
	double minMaxRew = numeric_limits<double>::infinity();
	double maxRew;
	for (sidx = 0; sidx < model->getNumberOfStates(); ++sidx) {

		maxRew = -numeric_limits<double>::infinity();
		model->updateNumberOfActions(sidx);
		for (aidx = 0; aidx < model->getNumberOfActions(); aidx++) {
			if (model->reward(sidx, aidx) > maxRew) {
				if (initPol){
					policy->assignPolicy(sidx,aidx); //argmax_a r(s,a)
				}
				maxRew = model->reward(sidx,aidx); //max_a r(s,a)
			}
		}

		if (maxRew < minMaxRew) {
			minMaxRew = maxRew;
		}
		if (maxRew > maxMaxRew) {
			maxMaxRew = maxRew;
		}

		if (initVal){
			valueVector->valueVector[sidx] = maxRew;
		}
	}

	//initialize v
	if (initVal && useDis){
		for (sidx = 0; sidx < model->getNumberOfStates(); ++sidx) {
			valueVector->valueVector[sidx] += model->getDiscount() / (1 - model->getDiscount()) * minMaxRew;
		}
	}

	//initialize  diffMin and diffMax
	diffMin = 0;
	diffMax = maxMaxRew - minMaxRew;
}


void ModifiedPolicyIteration::checkFinalValue() {
	//Check if final value vector is feasible
	//Note: This method only applies to the discounted reward optimality criterion
	
	if (useDis){ //check if discounted criterion selected
	
		//derive minimum and maximum rewards
		double minRew = numeric_limits<double>::infinity();
		double maxRew = -numeric_limits<double>::infinity();
		double r;
		int s,a;

		for (s = 0; s < model->getNumberOfStates(); s++) {
			model->updateNumberOfActions(s);
			for (a = 0; a < model->getNumberOfActions(); a++) {
				r = model->reward(s, a);
				if (r < minRew) {
					minRew = r;
				}
				if (r > maxRew) {
					maxRew = r;
				}
			}
		}
		if (minRew == -numeric_limits<double>::infinity()) {
			minRew = -1e6; // some large negative value
		}
		if (maxRew == numeric_limits<double>::infinity()) {
			maxRew = 1e6; // some large positive value
		}		
	
		//smallest and largest possible values in the value vector
		//are widened by the tolerance
		minRew = minRew / (1 - model->getDiscount()) - epsilon;
		maxRew = maxRew / (1 - model->getDiscount()) + epsilon;

		for (s = 0; s < model->getNumberOfStates(); ++s) {
			if (isnan(valueVector->valueVector[s]) || valueVector->valueVector[s] < minRew || valueVector->valueVector[s] > maxRew){
				valuesOutOfRange = true;
				outOfRangeState = s;
				outOfRangeValue = valueVector->valueVector[s];
				if (printStuff){
					cout << "NOT CONVERGED: Erroneous result in value vector at v[" << s << "] = " << valueVector->valueVector[s] << endl;
				}
				break;
			}
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

void ModifiedPolicyIteration::updateNorm(double &val) {
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

void ModifiedPolicyIteration::updateNorm(double val, double oldVal) {
	//as updateNorm(val), for a state whose old value is oldVal
	diff = val - oldVal;
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

void ModifiedPolicyIteration::setNorm(double newDiffMax, double newDiffMin) {
	//sets the norm from the largest and smallest difference of a whole sweep
	diffMax=newDiffMax;
	diffMin=newDiffMin;
	if (useStd) { //span norm
		norm = diffMax - diffMin;
	} else { //supremum norm: the largest absolute difference
		norm = (diffMax > -diffMin) ? diffMax : -diffMin;
	}
}