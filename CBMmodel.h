/* 
 * File:   CBMmodel.h
 * Author: Anders Reenberg Andersen and Jesper Fink Andersen
 *
 * Created on 18. september 2020, 12:00
 */

#ifndef CBMMODEL_H
#define CBMMODEL_H

#include <vector>
#include <string>

using namespace std;

class Model {
public:
    Model(int N,int L,double discount,string importProbPath = "");
    Model(const Model& orig);
    virtual ~Model();
    
    //general MDP parameters
    int N; //number of components
    int L; //failure limit of component
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
    int sNext; //next state to process
    double pNext; //transition probability to sNext
	vector<vector<int>> sidxMat; // (sidx,i)'th element contains s_i for state index sidx
	vector<vector<int>> aidxMat; // (aidx,i)'th element contains a_i for action index aidx
    
    //functions
    double reward(int, int);
    double transProb(int, int, int);
    void updateNext(int, int, int);
    int sFirst(int, int);

    //auxiliary functions
    int intPow(int, int);
    void importComponentProbs(string path); //import matrix of component transition probabilities
private:
};

#endif /* CBMMODEL_H */

