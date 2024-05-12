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

#include "ModelType.h"
#include "Policy.h"
#include "ValueVector.h"
#include "TBMmodel.h" //Time-based maintenance model
#include "CBMmodel.h" //Condition-based maintenance model
#include <vector>
#include <string>

using namespace std;

#ifndef MODIFIEDPOLICYITERATION_H
#define MODIFIEDPOLICYITERATION_H

class ModifiedPolicyIteration {
public:
    
    ModifiedPolicyIteration() {};
    ModifiedPolicyIteration(double eps=1e-3, string algorithm = "MPI", string update = "Standard",
            int parIterLim = 100, double SORrelaxation = 1.0, bool verbose=true, bool postProcessing=true,
             bool makeFinalCheck=true, bool parallel=true, bool genMDP=true);
    
    ModifiedPolicyIteration(const ModifiedPolicyIteration& orig);
    virtual ~ModifiedPolicyIteration();

    //other parameters
    double duration;
    int iter;
    bool converged;
    int polChanges; //count changes in policy in each iteration

    //methods
    void solve(ModelType * mdl, Policy * ply, ValueVector * vv);
    
private:

    //parameters
    double epsilon, diffMax, diffMin, diff, norm, tolerance, SORrelaxation, val, valBest, valSum, probSame, discount;
    int iterLim, parIter, parIterLim, PIparIterLim, sf, sidx, aidx, cidx, aBest, nJumps, nStates, nActions;
    bool useMPI, usePI, useVI, useStd, useGS, useSOR, initPol, initVal, printStuff, postProcessing, makeFinalCheck, genMDP, parallel;

    //pointer to model, policy, and value vector
    ModelType * model;
    Policy * policy;
    ValueVector * valueVector;
    
    //Pointers so we don't have to copy full vectors (for standard value function updates)
    vector<double> v2; //second value vector required when using standard updates
    vector<double> *vp; //pointer to last updated v
    vector<double> *vpOld; //pointer to old v
    vector<double> *vpTemp; //temporary pointer used when swapping vp and vpOld

    //methods
    void mainLoopModifiedPolicyIteration();
    void mainLoopValueIteration();
    
    void modifiedPolicyIterationGenMDP(); //MPI/PI for general MDP models (serial computation)
    void parModifiedPolicyIterationGenMDP(); //MPI/PI for general MDP models (parallel computation)
    void modifiedPolicyIterationSORGenMDP(); //MPI/PI with GS/SOR updates for general MDP models (serial computation)   
    void modifiedPolicyIteration(); //MPI/PI for built-in MDP models (serial computation)
    void modifiedPolicyIterationSOR(); //MPI/PI with GS/SOR updates for built-in MDP models (serial computation)   

    void valueIterationGenMDP(); //VI for general MDP models (serial computation)
    void parValueIterationGenMDP(); //VI for general MDP models (parallel computation)
    void valueIterationSORGenMDP(); //VI with GS/SOR updates for general MDP models (serial computation)
    void valueIteration(); //VI for built-in MDP models (serial computation)    
    void valueIterationSOR(); //VI with GS/SOR updates for built-in MDP models (serial computation)
    
    void initValue(); //initializes policy, v, and span
    void checkFinalValue();
    void print();
    
    //other methods
    void swapPointers(); //swaps vp and vpOld.
    void updateNorm(double &valBest); //updates diffMax, diffMin, and span/supNorm
    void computeNorm();
    
};

#endif /* MODIFIEDPOLICYITERATION_H */
