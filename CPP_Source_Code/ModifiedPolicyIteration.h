/*
* MIT License
*
* Copyright (c) 2026 Anders Reenberg Andersen and Jesper Fink Andersen
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
    ModifiedPolicyIteration(double eps=1e-3, string algorithm = "mpi", string update = "standard", string criterion = "discounted",
            int parIterLim = 100, double SORrelaxation = 1.0, bool verbose=true, bool postProcessing=true,
             bool makeFinalCheck=true, bool parallel=true);
    
    ModifiedPolicyIteration(const ModifiedPolicyIteration& orig);
    virtual ~ModifiedPolicyIteration();

    //other parameters
    double duration;
    int iter;
    bool converged; //no limit was reached and the final check (if made) found the values in range
    int polChanges; //count changes in policy in each iteration
    bool iterationLimitReached; //the main loop stopped at the iteration limit
    bool evaluationLimitReached; //a policy evaluation of PI stopped at its limit before reaching the tolerance
    bool valuesOutOfRange; //the final check found a value outside the possible range
    bool diverged; //the values became infinite or NaN, which ends the loops (a NaN norm passes their test)
    int outOfRangeState; //the first state with such a value (-1 if none)
    double outOfRangeValue; //and its value
    int getIterationLimit() { return iterLim; }

    //methods
    void solve(ModelType * mdl, Policy * ply, ValueVector * vv);
    
private:

    //parameters
    double epsilon, diffMax, diffMin, diff, norm, tolerance, SORrelaxation, val, valBest, valSum, probSame, discount;
    int iterLim, parIter, parIterLim, PIparIterLim, sf, sidx, aidx, cidx, aBest, nJumps, nStates, nActions;
    bool useMPI, usePI, useVI, useStd, useGS, useSOR, useDis, useAvg, initPol, initVal, printStuff, postProcessing, makeFinalCheck, parallel;

    //pointer to model, policy, and value vector
    ModelType * model;
    const CSRTables * tables; //the model's CSR tables, or nullptr for a model that computes its transitions on the fly
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
    
    //models with CSR tables (e.g. the general MDP model): the loops read the tables directly
    void modifiedPolicyIterationCSR(); //MPI/PI, standard updates (serial computation)
    void parModifiedPolicyIterationCSR(); //MPI/PI, standard updates (parallel computation)
    void modifiedPolicyIterationSORCSR(); //MPI/PI with GS/SOR updates (serial computation)
    void valueIterationCSR(); //VI, standard or Gauss-Seidel updates (serial computation)
    void parValueIterationCSR(); //VI, standard updates (parallel computation)
    void valueIterationSORCSR(); //VI with SOR updates (serial computation)

    //models that compute their transitions on the fly (the built-in TBM and CBM models)
    void modifiedPolicyIteration(); //MPI/PI (serial computation)
    void modifiedPolicyIterationSOR(); //MPI/PI with GS/SOR updates (serial computation)
    void valueIteration(); //VI (serial computation)
    void valueIterationSOR(); //VI with GS/SOR updates (serial computation)
    
    void initValue(); //initializes policy, v, and span
    void checkFinalValue();
    void print();
    
    //other methods
    void swapPointers(); //swaps vp and vpOld.
    void updateNorm(double &valBest); //updates diffMax, diffMin, and span/supNorm
    void updateNorm(double val, double oldVal); //the same, with the old value of the state given
    void setNorm(double newDiffMax, double newDiffMin); //sets diffMax, diffMin, and span/supNorm
    
};

#endif /* MODIFIEDPOLICYITERATION_H */
