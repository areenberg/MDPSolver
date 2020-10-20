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


#ifndef CBMMODEL_H
#define CBMMODEL_H

#include <vector>
#include <string>

using namespace std;

class CBMmodel {
public:
    CBMmodel(int Ninput,int Linput,double discountInput,
            string importProbPath = "");
    CBMmodel(const CBMmodel& orig);
    virtual ~CBMmodel();

    //general MDP parameters
    int N;
    int L;
    double discount;
    int numberOfStates;
    int numberOfActions;
    vector<int> policy;
    //transition and reward parameters
    double cp; //preventive replacement cost
    double cc; //corrective replacement cost
    double cs; //setup cost
    double p; //system failure penalty
    int kN; // k-out-of-N system, meaning system works iff k components work
    bool importProbs;
    vector<vector<double>> pCompMat; //component transition probs.
	vector<vector<double>> pFailCompMat; //sum of element in pCompMat
    //auxiliary variables
    int nextState; //next state to process
    double psj; //transition probability
	int s_i, a_i, j_i;
	vector<vector<int>> sidxMat; // (sidx,i)'th element contains s_i for state index sidx
	vector<vector<int>> aidxMat; // (aidx,i)'th element contains a_i for action index aidx
    // functions
    double reward(int, int);
    double transProb(int, int, int);
    void updateNextState(int, int, int);
    void updateTransProbNextState(int, int, int);
    int postDecisionIdx(int, int);
    int intPow(int, int);
    void importComponentProbs(string path);
private:
};

#endif /* MODEL_H */
