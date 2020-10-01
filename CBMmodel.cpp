/* 
 * File:   CBMmodel.cpp
 * Author: Anders Reenberg Andersen and Jesper Fink Andersen
 * 
 * Created on 18. september 2020, 12:00
 */

#include <iostream>
#include <fstream> //to import component probabilities
#include <string>  //
#include <sstream> // 
#include <math.h>
#include <assert.h>
#include <exception> //for exiting if probabilities cannot be loaded
#include <algorithm> //for min and max function
#include "CBMmodel.h"

using namespace std;

Model::Model(int N, int L, double discount, string importProbPath): //default constructor
    N(N),
    L(L),
    discount(discount),
    numberOfStates(intPow(L+1,N)), //unsigned INT_MAX is 2147483647
    numberOfActions(intPow(2,N)),
    cp(-5),
    cc(-11),
    cs(-4),
    p(-300),
	kN(max(1,N-1)),
    importProbs(!importProbPath.empty()),    
    pCompMat(N,vector<double>(L+1)),
	pFailCompMat(N, vector<double>(L + 1)),
    policy(numberOfStates,0),
	sidxMat(numberOfStates, vector<int>(N)),
	aidxMat(numberOfActions, vector<int>(N))
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

Model::Model(const Model& orig) {
}

Model::~Model() {
}

//class functions
double Model::reward(int sidx,int aidx) {
    //reward function
	int s_i, a_i;
    double r = 0;
    bool set_up = false;
    int fail_count = 0;
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
    return r;
}

double Model::transProb(int sidx, int aidx, int jidx) {
	//transition probability function

	int step, s_i, a_i, j_i;
	double prob = 1;
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
	pNext = prob; //set stored transition probability
	return prob;
}

void Model::updateNext(int sidx, int aidx, int jidx) {
	//updates pNext and sNext, which are assumed to match. 
	//That is, input jidx should be sNext.

	int step, s_i, a_i, j_i;
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
			sNext += intPow(L + 1, i); //increment i'th component by one
			if (j_i + 1 < L) {
				pNext *= pCompMat[i][step + 1] / pCompMat[i][step]; // still lower than L after increment
			} else {
				if (a_i == 0) {
					pNext *= pFailCompMat[i][s_i] / pCompMat[i][step];
				} else {
					pNext *= pCompMat[i][j_i + 1] / pCompMat[i][step]; //going from 0 to L here
				}
			}
			break; //we are done
		} else {
			sNext -= step * intPow(L + 1, i); //reset back to s_i or 0
			if (a_i == 0) { //went from s_i to L
				pNext /= pFailCompMat[i][s_i];
				if (s_i < L) {
					pNext *= pCompMat[i][0];
				}
			} else {
				pNext *= pCompMat[i][0] / pCompMat[i][L];
			}
		}
	}
}

int Model::sFirst(int sidx, int aidx) {
	//state index right after replacement, which is
	//assumed instantaneous so components are set to age 0
	int pdidx = sidx;
	int s_i, a_i;
	for (int i = 0; i<N; ++i) {
		s_i = sidxMat[sidx][i];
		a_i = aidxMat[aidx][i];
		if (a_i == 1) {
			pdidx -= (s_i)*intPow(L + 1, i); // sets it to 0
		}
	}
	sNext = pdidx; //store as the "first" next state
	return pdidx;
}

int Model::intPow(int a, int b) {
    int i = 1;
    for(int j = 1; j <= b; ++j) i *= a;
    return i;
}

void Model::importComponentProbs(string path) {
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
            ++j;
        }
        ++i;
    }
    inputFile.close();
}

