/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Model.cpp
 * Author: jfan
 * 
 * Created on 8. november 2019, 08:56
 */

//#include <iostream>
#include <fstream> //to import component probabilities
#include <string>  //
#include <sstream> // 
#include <math.h> //for the fabs
#include <assert.h>
#include <limits>
#include <random> //for initializing rj_vec and lj_vec
#include "Model.h"

using namespace std;

Model::Model(int N, int L, double discount): 
	N(N),
	L(L),
	discount(discount),
	numberOfStates(intPow(L + 1, N)),
	numberOfActions(intPow(2, N)),
	rj(-10),
	Rs(-10),
	Rf(-(double)5 * N),
	penalty(-1e6),
	f(0.1),
	fmin(0.01),
	fhat(0.1),
	policy(numberOfStates, 0),
	failOddsVec(N,0),
	sidxMat(numberOfStates, vector<int>(N)),
	aidxMat(numberOfActions, vector<int>(N)),
	sidxSumMat(numberOfStates, 0)
{
	int sidxTemp, sm, aidxTemp;
	//initialize sidxMat 
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

Model::Model(const Model& orig) {
}

Model::~Model() {
}

//class functions
double Model::reward(int sidx,int aidx) {
	//reward function
	int s_i, a_i;
    double r = 0;
    bool setUp = false;
	double noFailProb=1;
	bool payPenalty = false;
	
    for(int i = 0; i < N; ++i) {
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];
        if (a_i==0) { // no replacement
            if(s_i==0) {
				payPenalty = true;
			} else if (s_i > 1) {
				// probability of not failing
				if (N > 1) {
					noFailProb *= 1.0 - (f - (f - fmin)*(s_i - 1.0) / (L - 1.0) + fhat * ((N - 1.0)*L - (sidxSumMat[sidx] - s_i)) / ((N - 1.0)*L));
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

double Model::transProb(int sidx, int aidx, int jidx) {
	//probability of transitioning to state j given we are in state s and take action a
	int s_i, j_i, a_i;
	double prob = 1;
	double failProb;


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

void Model::updateTransProbNextState(int sidx, int aidx, int jidx) {
	//updates psj and nextState. Assumes that transProb(sidx,aidx,pdidx) has been run,
	//such that failOddsVec is up to date.
	int s_i, j_i, a_i;
	
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

int Model::postDecisionIdx(int sidx, int aidx) {
	//returns state index after replacements
    //replaced components reset to L 
    //other components age by 1
	int s_i, j_i, a_i;
    
    int pdidx = sidx;
    for (int i=0; i<N; ++i) {
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];
        if (a_i==1) {
            pdidx += (L-s_i)*intPow(L+1,i); // sets it to L
        } else if (0<s_i) {
            pdidx -= intPow(L+1,i); // working components age by 1
        }
    }
	nextState = pdidx; //store as the "first" next state
    return pdidx;
}

int Model::intPow(int a, int b) {
    int i = 1;
    for(int j = 1; j <= b; ++j) i *= a;
    return i;
}
