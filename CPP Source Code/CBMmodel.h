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


#ifndef CBMMODEL_H
#define CBMMODEL_H

#include "ModelType.h"
#include <vector>
#include <string>

using namespace std;

class CBMmodel : public ModelType{
public:
    
    //constructor and destructor
    CBMmodel() {}; //dummy-constructor
    CBMmodel(double discount=0.99,
            int components=2,
            int stages=10,
            string importProbPath="pCompMat.csv",
            double preventiveCost=-5,
            double correctiveCost=-11,
            double setupCost=-4,
            double failurePenalty=-300,
            int kOfN=-1);
    CBMmodel(double discount=0.99,
            int components=2,
            int stages=10,
            vector<vector<double>> pcm={},
            double preventiveCost=-5,
            double correctiveCost=-11,
            double setupCost=-4,
            double failurePenalty=-300,
            int kOfN=-1);
    CBMmodel(const CBMmodel& orig);
    virtual ~CBMmodel();

    //general MDP parameters
    int N;
    int L;
    double discount;
    int numberOfStates;
    int numberOfActions;
    //vector<int> policy;
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
    double psj; //transition probability from state s to j
	int s_i, a_i, j_i;
	vector<vector<int>> sidxMat; // (sidx,i)'th element contains s_i for state index sidx
	vector<vector<int>> aidxMat; // (aidx,i)'th element contains a_i for action index aidx
    
    // METHODS
        
    //GENERIC METHODS    
    double * reward(int &sidx, int &aidx) override;
    double * transProb(int &sidx, int &aidx, int &jidx) override;
    void updateNextState(int &sidx, int &aidx, int &jidx) override;
    int * postDecisionIdx(int &sidx, int &aidx) override;
    double * getDiscount() override;
    int * getNumberOfStates() override;
    void updateNumberOfActions(int &sidx) override;
    int * getNumberOfActions() override;
    int * getNextState() override;
    double * getPsj() override;
    
    //SPECIAL METHODS
    void updateTransProbNextState(int, int, int);
    int intPow(int, int);
    void importComponentProbs(string path);
    
private:

    double r,prob;
    bool set_up,done;
    int fail_count,step,pdidx;

};

#endif /* CBMMODEL_H */
