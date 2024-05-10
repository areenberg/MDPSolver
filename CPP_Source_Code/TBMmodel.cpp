/*
* MIT License
*
* Copyright (c) 2024 Anders Reenberg Andersen and Jesper Fink Andersen
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


#include "TBMmodel.h"
//#include "Policy.h"
#include <iostream>

using namespace std;

TBMmodel::TBMmodel(double discount,
	int components,
    int stages,
    double replacementCost,
    double setupCost,
    double unexpectedFailureCost,
    double expiredNotFixedCost,
    double failureProb,
    double failureProbMin,
    double failureProbHat):
	N(components),
	L(stages-1),
	discount(discount),
	numberOfStates(intPow(stages, components)),
	numberOfActions(intPow(2, components)),
	rj(replacementCost),
	Rs(setupCost),
	Rf(unexpectedFailureCost),
	penalty(expiredNotFixedCost),
	f(failureProb),
	fmin(failureProbMin),
	fhat(failureProbHat),
	failOddsVec(components,0),
	sidxMat(numberOfStates, vector<int>(components)),
	aidxMat(numberOfActions, vector<int>(components)),
	sidxSumMat(numberOfStates, 0)
{
	//initialize sidxMat
	int s_i, a_i, sidxTemp, sm, aidxTemp;
	for (int sidx = 0; sidx < numberOfStates; ++sidx) {
		sidxTemp = sidx;
		sm = 0;
		for (int i = 0; i < N; ++i) {
			s_i = sidxTemp % (L + 1);
			sidxMat[sidx][i] = s_i;
			sm += s_i;
			sidxTemp /= (L + 1);
		}
		sidxSumMat[sidx] = sm;
	}

	//initialize aidxMat
	for (int aidx = 0; aidx < numberOfActions; ++aidx) {
		aidxTemp = aidx;
		for (int i = 0; i < N; ++i) {
			a_i = aidxTemp % 2;
			aidxMat[aidx][i] = a_i;
			aidxTemp /= 2;
		}
	}
}

TBMmodel::TBMmodel(const TBMmodel& orig) {
}

TBMmodel::~TBMmodel() {
}

//class functions
double TBMmodel::reward(int &sidx,int &aidx) {
	//reward function
	//int s_i, a_i;
	r = 0;
    setUp = false;
	noFailProb=1;
	payPenalty = false;

    for(int i = 0; i < N; ++i) {
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];
        if (a_i==0) { // no replacement
            if(s_i==0) {
				payPenalty = true;
			} else if (s_i > 1) {
				// probability of not failing
				if (N > 1) {
					noFailProb *= 1.0 - (f - (f - fmin)*(s_i - 1.0) / (L - 1.0)
						+ fhat * ((N - 1.0)*L - (sidxSumMat[sidx] - s_i)) / ((N - 1.0)*L));
				} else {
					noFailProb *= 1.0 - (f - (f - fmin)*(s_i - 1.0) / (L - 1.0));
				}
			}
        } else { //replacement
            r += rj;
            setUp = true;
        }
    }
    r += setUp*Rs + (1-noFailProb)*Rf + payPenalty*penalty;
    return r;
}

double TBMmodel::transProb(int &sidx, int &aidx, int &jidx) {
	//probability of transitioning to state j given we are in state s and take action a
	//int s_i, j_i, a_i;
	prob = 1;
	//double failProb;

	for (int i = 0; i<N; ++i) {
		j_i = sidxMat[jidx][i];
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];

		if (a_i == 0) {
			if (s_i > 1) {
				if (N>1) {
					failProb = f - (f - fmin)*(s_i - 1.0) / (L - 1.0) + fhat * ((N - 1.0)*L - (sidxSumMat[sidx] - s_i)) / ((N - 1.0)*L);
				} else {
					failProb = f - (f - fmin)*(s_i - 1.0) / (L - 1.0);
				}
				if (j_i == 0) { //failure
					prob *= failProb;
				} else if (j_i == (s_i - 1)) { //no failure
					prob *= 1.0 - failProb;
				} else { //impossible transition
					prob *= 0;
				}
				failOddsVec[i] = failProb / (1 - failProb); //store for faster computations of other reachable states
			} else if ((s_i == 0 && j_i != 0) || (s_i == 1 && j_i != 0)) { //impossible transition
				prob *= 0;
			}
		} else if (j_i != L) { //impossible transition
			prob *= 0;
		}
	}
	psj = prob; //store transition probability
	return prob;
}

void TBMmodel::updateNextState(int &sidx, int &aidx, int &jidx) {
	//updates pNext and sNext. Assumes that transProb(sidx,aidx,pdidx) has been run,
	//such that failOddsVec is up to date.
	//int s_i, j_i, a_i;

	for (int i = 0; i<N; ++i) {
		j_i = sidxMat[jidx][i];
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];
		if (a_i==0 && 0<j_i && s_i != 0) { //non-replacements, working component
			if ((j_i - s_i) == -1) {
				nextState -= j_i * intPow(L + 1, i); //decrease to 0  (failure)
				psj *= failOddsVec[i]; //failOdds=failProb/(1-failProb)
			}
			break; //the remaining components don't change
		} else if (a_i==0 && s_i > 1) { //only if i'th component was able to fail
			nextState -= (j_i - (s_i - 1))*intPow(L + 1, i); //reset back to s_i-1 (not failed)
			psj /= failOddsVec[i]; //failOdds=(1-failProb)/failProb
		}
	}
}

int * TBMmodel::postDecisionIdx(int &s, int &a) {
	//returns state index after replacements
    //replaced components reset to L
    //other components age by 1
	//int s_i, a_i;

    sf = s;
    for (int i=0; i<N; ++i) {
		s_i = sidxMat[s][i];
		a_i = aidxMat[a][i];
        if (a_i==1) {
            sf += (L-s_i)*intPow(L+1,i); // sets it to L
        } else if (0<s_i) {
            sf -= intPow(L+1,i); // working components age by 1
        }
    }
	nextState = sf; //store as the "first" next state
    return &sf;
}

int TBMmodel::intPow(int a, int b) {
    int i = 1;
    for(int j = 1; j <= b; ++j) i *= a;
    return i;
}

double TBMmodel::getDiscount(){
    return discount;
}

int TBMmodel::getNumberOfStates(){
    return numberOfStates;
}

void TBMmodel::updateNumberOfActions(int &sidx){	
}

int TBMmodel::getNumberOfActions(){
    return numberOfActions;
}

int * TBMmodel::getNextState(){
    return &nextState;
}

double * TBMmodel::getPsj(){
    return &psj;
}

int TBMmodel::getColumnIdx(int &sidx, int &aidx, int &cidx){
	return 0;
}

int TBMmodel::getNumberOfJumps(int &sidx, int &aidx){
	return 0;
}

int TBMmodel::getNumberOfActions(int &sidx){
	return 0;
}

//int TBMmodel::getPolicy(int sidx){
//    return policy[sidx];
//}
//
//void TBMmodel::assignPolicy(int sidx, int action){
//    policy[sidx] = action;
//}