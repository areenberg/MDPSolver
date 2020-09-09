/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Model.h
 * Author: jfan
 *
 * Created on 8. november 2019, 08:56
 */

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>

using namespace std;

class Model {
public:
    Model(int Ninput,int Linput,double discountInput,
            string importProbPath = "");
    Model(const Model& orig);
    virtual ~Model();
    
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

