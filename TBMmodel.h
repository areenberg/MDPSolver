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


#ifndef TBMMODEL_H
#define TBMMODEL_H

#include <vector>

using namespace std;

class TBMmodel {
public:
    //contsructor and destructer
    //dummy constructor (not included in cpp-file) 
    TBMmodel() {};
    TBMmodel(int N,int L,double discount);
    TBMmodel(const TBMmodel& orig);
    virtual ~TBMmodel();

    //general MDP parameters
    int N; //number of components
    int L; //maximum lifetime of components
    double discount;
	int numberOfStates;
    int numberOfActions;
    vector<int> policy;

    //transition and reward parameters
    double rj; //replacement cost
    double Rs; //setup cost
    double Rf; //component (unexpected) failure extra cost.
    double penalty; // penalty if expired component is not fixed (should be very large to avoid this)
    double f; // failure probability parameters
    double fmin; // -||-
    double fhat; // -||-

    //auxiliary variables
    int nextState; //int sNext; //next state to process
    double psj; //double pNext; //transition probability from state s to j
	vector<double> failOddsVec; // probability of failing divided by probability of not failing
	vector<vector<int>> sidxMat; // (sidx,i)'th element contains s_i for state index sidx
	vector<vector<int>> aidxMat; // (aidx,i)'th element contains a_i for action index aidx
	vector<int> sidxSumMat; // sidx'th element contains the sum of component states

    //methods
    double reward(int, int);
    double transProb(int, int, int);
    void updateNextState(int, int, int); //void updateNext(int, int, int);
    int postDecisionIdx(int, int); //int sFirst(int, int);
    

	//auxiliary methods
    int intPow(int, int);
private:
};

#endif /* TBMMODEL_H */
