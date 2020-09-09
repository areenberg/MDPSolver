/* 
 * File:   TBMmodel.h
 * Author: Anders Reenberg Andersen and Jesper Fink Andersen
 *
 * Created on 8. july 2020, 10:00
 */

#ifndef TBMMODEL_H
#define TBMMODEL_H

#include <vector>
#include <string>

using namespace std;

class Model {
public:
    Model(int N,int L,double discount, bool nonIdentical=false, unsigned int seed=1);
    Model(const Model& orig);
    virtual ~Model();
    
    //general MDP parameters
    int N; //number of components
    int L; //expected lifetime of components
    double discount;
	int numberOfStates;
    int numberOfActions;	
    vector<int> policy;
	
    //transition and reward parameters
    double rj; //replacement cost
    double Rs; //fixed setup
    double Rf; //component (unexpected) failure extra cost.
    double penalty; // should be infinite
    double f; // failure probability parameters
    double fmin; // -
    double fhat; // -
	
    //auxiliary variables
    int nextState; //next state to process
    double psj; //transition probability from state s to j
	vector<double> failOddsVec; // The probability 
	vector<vector<int>> sidxMat; // (sidx,i)'th element contains s_i for state index sidx
	vector<vector<int>> aidxMat; // (aidx,i)'th element contains a_i for action index aidx
	vector<int> sidxSumMat; // sidx'th element contains the sum of component states

    // functions
    double reward(int, int);
    double transProb(int, int, int);
    void updateTransProbNextState(int, int, int);
    int postDecisionIdx(int, int);

	//non-identical component functions
	double rewardNonIdentical(int, int);
	double transProbNonIdentical(int, int, int);
	void updateNextStateNonIdentical(int, int, int);
	void updateTransProbNextStateNonIdentical(int, int, int);
	int postDecisionIdxNonIdentical(int, int);

	//auxiliary functions
    int intPow(int, int);
private:
};

#endif /* TBMMODEL_H */

