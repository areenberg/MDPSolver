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


#include <iostream>
#include <fstream> //to import component probabilities
#include <string>  //
#include <sstream> //
#include <math.h>
#include <assert.h>
#include <exception> //for exiting if probabilities cannot be loaded
#include <algorithm>
#include <valarray> //for min and max function
#include "CBMmodel.h"

using namespace std;

CBMmodel::CBMmodel(int components, int stages, double discount, string importProbPath): //default constructor
    N(components),
    L(stages-1),
    discount(discount),
    numberOfStates(intPow(stages,components)), //unsigned INT_MAX is 2147483647
    numberOfActions(intPow(2,components)),
    cp(-5),
    cc(-11),
    cs(-4),
    p(-300),
	kN(max(1,components-1)),
    importProbs(!importProbPath.empty()),
    pCompMat(components,vector<double>(stages)),
	pFailCompMat(components, vector<double>(stages)),
	sidxMat(numberOfStates, vector<int>(components)),
	aidxMat(numberOfActions, vector<int>(components))
{
    
    //import component probabilities
    if (importProbs) {
        importComponentProbs(importProbPath);
    }
	// calculate tail probabilities
	for (int i = 0; i < N; ++i) {
		pFailCompMat[i][0] = pCompMat[i][L];
		for (int j = 1; j <= L; ++j) {
			pFailCompMat[i][j] = pFailCompMat[i][j - 1] + pCompMat[i][L-j];
		}
	}
	//initialize sidxMat
	int sidx_temp,s_i;
	for (int sidx = 0; sidx < numberOfStates; ++sidx) {
		sidx_temp = sidx;
		for (int i = 0; i < N; ++i) {
			s_i = sidx_temp % (L+1);
			sidxMat[sidx][i] = s_i;
			sidx_temp /= (L+1);
		}
	}
	//initialize aidxMat
	int aidx_temp, a_i;
	for (int aidx = 0; aidx < numberOfActions; ++aidx) {
		aidx_temp = aidx;
		for (int i = 0; i < N; ++i) {
			a_i = aidx_temp % 2;
			aidxMat[aidx][i] = a_i;
			aidx_temp /= 2;
		}
	}
}

CBMmodel::CBMmodel(int components, int stages, double discount, vector<vector<double>> pcm): //default constructor
    N(components),
    L(stages-1),
    discount(discount),
    numberOfStates(intPow(stages,components)), //unsigned INT_MAX is 2147483647
    numberOfActions(intPow(2,components)),
    cp(-5),
    cc(-11),
    cs(-4),
    p(-300),
	kN(max(1,components-1)),
    pCompMat(pcm),
	pFailCompMat(components, vector<double>(stages)),
	sidxMat(numberOfStates, vector<int>(components)),
	aidxMat(numberOfActions, vector<int>(components))
{
    
	// calculate tail probabilities
	for (int i = 0; i < N; ++i) {
		pFailCompMat[i][0] = pCompMat[i][L];
		for (int j = 1; j <= L; ++j) {
			pFailCompMat[i][j] = pFailCompMat[i][j - 1] + pCompMat[i][L-j];
		}
	}
	//initialize sidxMat
	int sidx_temp,s_i;
	for (int sidx = 0; sidx < numberOfStates; ++sidx) {
		sidx_temp = sidx;
		for (int i = 0; i < N; ++i) {
			s_i = sidx_temp % (L+1);
			sidxMat[sidx][i] = s_i;
			sidx_temp /= (L+1);
		}
	}
	//initialize aidxMat
	int aidx_temp, a_i;
	for (int aidx = 0; aidx < numberOfActions; ++aidx) {
		aidx_temp = aidx;
		for (int i = 0; i < N; ++i) {
			a_i = aidx_temp % 2;
			aidxMat[aidx][i] = a_i;
			aidx_temp /= 2;
		}
	}
}


CBMmodel::CBMmodel(const CBMmodel& orig) {
}

CBMmodel::~CBMmodel() {
}

//class functions
double * CBMmodel::reward(int &sidx, int &aidx) {
    //reward function
    r = 0;
    set_up = false;
    fail_count = 0;
    for(int i = 0; i < N; ++i) {
		s_i = sidxMat[sidx][i];//i'th component state
		a_i = aidxMat[aidx][i];
        if (a_i==1) {
            set_up = true;
            if (s_i==L) {
                ++fail_count;
                r += cc;
            } else {
                r += cp;
            }
        } else if (s_i==L) {
            ++fail_count;
        }
    }
    r += set_up*cs + ((N-fail_count) < kN)*p;
    return &r;
}

double * CBMmodel::transProb(int &sidx, int &aidx, int &jidx) {
	//transition probability function

	//int step;
	prob = 1;
	for (int i = 0; i<N; ++i) {
		j_i = sidxMat[jidx][i]; //i'th component state
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];
		if (a_i == 0 && j_i<s_i) { // impossible transitions
			prob *= 0;
		} else if (a_i == 0 && j_i >= s_i) { //no replacement
			step = j_i - s_i;
			if (j_i < L) {
				prob *= pCompMat[i][step];
			} else {
				prob *= pFailCompMat[i][s_i];
			}
		} else { //replacement
			prob *= pCompMat[i][j_i];
		}
	}
	psj = prob; //set stored transition probability
	return &prob;
}

void CBMmodel::updateTransProbNextState(int sidx, int aidx, int jidx) {
	//void CBMmodel::updateTransProbNextStateOptimized(int sidx, int aidx, int jidx) {
	//updates psj and nextState, which are assumed to match.
	//That is, input jidx should be nextState.

	int step;
	for (int i = 0; i<N; ++i) {
		j_i = sidxMat[jidx][i]; //i'th component state
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];

		//assert(a_i == 1 || (a_i == 0 && j_i >= s_i));//check valid transition

		if (a_i == 0) {
			step = j_i - s_i;
		} else {
			step = j_i;
		}

		if (j_i<L) {
			nextState += intPow(L + 1, i); //increment i'th component by one
			//psj /= pCompMat[i][step]; //psj should be divided by what was previously used to calc component transition
			if (j_i + 1 < L) {
				psj *= pCompMat[i][step + 1] / pCompMat[i][step]; // still lower than L after increment
			} else {
				if (a_i == 0) {
					psj *= pFailCompMat[i][s_i] / pCompMat[i][step];
				} else {
					psj *= pCompMat[i][j_i + 1] / pCompMat[i][step]; //going from 0 to L here
				}
			}
			break; //we are done
		} else {
			nextState -= step * intPow(L + 1, i); //reset back to s_i or 0
			//here j_i=L so psj was formely multiplied with a fail probability
			if (a_i == 0) { //went from s_i to L
				psj /= pFailCompMat[i][s_i];
				if (s_i < L) {
					psj *= pCompMat[i][0];
				}
			} else {
				psj *= pCompMat[i][0] / pCompMat[i][L];
			}
		}
	}
}

int * CBMmodel::postDecisionIdx(int &sidx, int &aidx) {
	//int CBMmodel::postDecisionIdxOptimized(int sidx, int aidx) {
	//state index right after replacement, which is
	//assumed instantaneous so components are set to age 0
	pdidx = sidx;

	for (int i = 0; i<N; ++i) {
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];
		if (a_i == 1) {
			pdidx -= (s_i)*intPow(L + 1, i); // sets it to 0
		}
	}
	nextState = pdidx; //store as the "first" next state
	return &pdidx;
}


void CBMmodel::updateNextState(int &sidx, int &aidx, int &jidx) {
	//increment one component's deterioration level.
	if (jidx != -1) {
		nextState = jidx;
	} else {
		jidx = nextState; // default jidx value is current nextState
	}
	done = false;
	for (int i = 0; i<N; ++i) {
		j_i = sidxMat[jidx][i]; //i'th component state
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];

		if (!done && j_i<L) { //non-replacements
			nextState += intPow(L + 1, i);
			done = true;
		} else if (!done) {
			if (a_i == 0) {
				nextState -= (j_i - s_i)*intPow(L + 1, i); //reset back to s_i
			} else {
				nextState -= (j_i)*intPow(L + 1, i); //reset back to 0
			}
		}
	}
}


int CBMmodel::intPow(int a, int b) {
    int i = 1;
    for(int j = 1; j <= b; ++j) i *= a;
    return i;
}

void CBMmodel::importComponentProbs(string path) {
    string line;
    ifstream inputFile (path);
    try {
        if(!inputFile.is_open()) {
            throw path;
        }
    }
    catch (string path) {
        cout << "Exception: UNABLE TO OPEN FILE: " + path + "\n";
        exit(EXIT_FAILURE);
    }

    int i = 0;
    int j = 0;
    string element;
    while( getline(inputFile, line) ) {
        istringstream lineStream(line);
        // read every element from the line that is separated by commas
        // and put it into pCompMat
        j = 0;
        while( getline(lineStream, element, ',') ){
            pCompMat [i][j] = stod(element);
            //cout << i << " " << j << " " << element << endl;
            ++j;
        }
        ++i;
    }
    inputFile.close();
}

double * CBMmodel::getDiscount(){
    return &discount;
}

int * CBMmodel::getNumberOfStates(){
    return &numberOfStates;
}

void CBMmodel::updateNumberOfActions(int &sidx){	
}

int * CBMmodel::getNumberOfActions(){
    return &numberOfActions;
}      

int * CBMmodel::getNextState(){
    return &nextState;
}

double * CBMmodel::getPsj(){
    return &psj;
}

//int CBMmodel::getPolicy(int sidx){
//    return policy[sidx];
//}
//
//void CBMmodel::assignPolicy(int sidx, int action){
//    policy[sidx] = action;
//}
