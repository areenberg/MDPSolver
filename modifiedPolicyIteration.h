/* 
 * File:   modifiedPolicyIteration.h
 * Author: Anders Reenberg Andersen and Jesper Fink Andersen
 *
 * Created on 20. september 2020, 12:00
 */

//model class
//#include "TBMmodel.h"
#include "CBMmodel.h"

#include <vector>
#include <string>

using namespace std;

#ifndef MODIFIEDPOLICYITERATION_H
#define MODIFIEDPOLICYITERATION_H

class modifiedPolicyIteration {
public:
    //constructor and destructor
    modifiedPolicyIteration(Model& model, double eps, string algorithm = "MPI", string update = "Standard",
        int parIterLim = 100, double SORrelaxation = 1.0);
    modifiedPolicyIteration(const modifiedPolicyIteration& orig);
    virtual ~modifiedPolicyIteration();

    //value vector
    vector<double> v;
	
    //other parameters
    double duration;
	int iter;
	bool converged;
    int polChanges; //count changes in policy in each iteration
    
    //medthods
    void solve(Model& model);
    
private:

    //parameters
    double epsilon, diffMax, diffMin, norm, tolerance, discount, SORrelaxation;
    int iterLim, parIter, parIterLim, PIparIterLim;
    bool useMPI, usePI, useVI, useStd, useGS, useSOR, printStuff;

	//Pointers so we don't have to copy full vectors (for standard value function updates)
	vector<double> v2; //second value vector required when using standard updates
	vector<double> *vp; //pointer to last updated v
	vector<double> *vpOld; //pointer to old v
	vector<double> *vpTemp; //temporary pointer used when swapping vp and vpOld

    //methods
	void improvePolicy(Model& model);
	void partialEvaluation(Model& model);
    void improvePolicySOR(Model& model);
    void partialEvaluationSOR(Model& model);
	void initValue(Model& model); //initializes policy, v, and span
	void swapPointers(); //swaps vp and vpOld.
	void updateNorm(int s, double valBest); //updates diffMax, diffMin, and span/supNorm
    void checkFinalValue(Model& model);
};

#endif /* MODIFIEDPOLICYITERATION_H */

