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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <string>
#include <iostream>

#include "ModelType.h" //Generic model class
#include "ModifiedPolicyIteration.h" //The solver
#include "TransitionMatrix.h" //Stores transition matrix in general MDP model
#include "Rewards.h" //Stores rewards in general MDP model

//MODEL TYPES
#include "GeneralMDPmodel.h" //General MDP model
#include "TBMmodel.h" //Time-based maintenance model
#include "CBMmodel.h" //Condition-based maintenance model

using namespace std;
namespace py = pybind11;

using namespace std;

#ifndef MODULEINTERFACE_H
#define MODULEINTERFACE_H

class ModuleInterface {
public:
    
    ModuleInterface();
    ModuleInterface(const ModuleInterface& orig);
    virtual ~ModuleInterface();
    

    //---------------------------------------
    //  DATA STRUCTURES
    //---------------------------------------   

    //problem settings
    struct Problem{
        string problemType;
        double discount;

        //policy and value vector
        Policy policy;
        ValueVector valueVector;

        //transition matrix and rewards (only for general MDP model)
        TransitionMatrix tranMat;
        Rewards rewards;
    
        //only for the TBM/CBM models
        int components;
        int stages;
        double replacementCost;
        double setupCost;
        double preventiveCost;
        double correctiveCost;
        double failurePenalty;
        double unexpectedFailureCost;
        double expiredNotFixedCost;
        double failureProb;
        double failureProbMin;
        double failureProbHat;
        int kOfN;
        vector<vector<double>> pCompMat;
    } problem;


    //solver settings
    struct Settings{
        string algorithm;
        double tolerance;
        string update;
        string criterion;
        int parIterLim;
        double SORrelaxation;
        bool verbose;
        bool postProcessing;
        bool makeFinalCheck;
        bool parallel;
    } settings;


    //results
    struct Results{
        //duration (runtime) in milliseconds
        double duration=0;
        //number of iterations (policy improvement steps; every update is one with VI)
        int iterations=0;
        //whether the last solve converged, and if not, why
        bool converged=false;
        bool iterationLimitReached=false;
        bool evaluationLimitReached=false;
        bool valuesOutOfRange=false;
        bool diverged=false;
        int outOfRangeState=-1;
        double outOfRangeValue=0.0;
        int iterationLimit=0;
    } results;

    
    //---------------------------------------
    //  METHODS
    //---------------------------------------

    //------ problem selection ------

    //the general MDP problem
    //(each format but the files is a list or a NumPy array; see mdp() in ModuleInterface.cpp)
    void mdp(double discount,
    py::object rewards, //option1: complete reward list 
    py::object rewardsElementwise, //option2: rewards where each row is an element and columns specify sidx,aidx,reward
    string rewardsFromFile, //option3: rewards are loaded from a file
    py::object tranMatWithZeros, //option1: complete transition mat incl. zeros
    py::object tranMatElementwise, //option2: tran mat where each row is a non-zero element and columns specify sidx,aidx,jidx,prob
    py::object tranMatProbs, //option3a: transition mat non-zero probabilities
    py::object tranMatColumns, //option3b: transition mat column indices 
    string tranMatFromFile); //option4: transition mat is loaded from a file

    // ------ pre-defined MDP problems ------  
    void tbm(double discount, //select TBM problem
        int components,
        int stages,
        double replacementCost,
        double setupCost,
        double unexpectedFailureCost,
        double expiredNotFixedCost,
        double failureProb,
        double failureProbMin,
        double failureProbHat); 
    void cbm(double discount, //select CBM problem
        int components,
        int stages,
        py::list pCompMat,
        double preventiveCost,
        double correctiveCost,
        double setupCost,
        double failurePenalty,
        int kOfN);

    //-------------------------------

    void solve(string algorithm="mpi", //solves the problem
     double tolerance=1e-3,
     string update="standard",
     string criterion="discounted",
     int parIterLim=100,
     double SORrelaxation=1.0,
     py::list initPolicy=py::list(),
     py::list initValueVector=py::list(),
     bool verbose=false,
     bool postProcessing=true,
     bool makeFinalCheck=true,
     bool parallel=true); 
    
    //-------------------------------

    //------ output methods ------
    
    void printPolicy(); //prints the policy
    void printValueVector(); //prints the value vector
    int getAction(int sidx); //returns the action index (from the optimized policy) associated with the current state
    double getValue(int sidx); //returns the value (from the optimized policy) associated with the current state
    double getRuntime(); //returns the runtime in milliseconds
    int getIterations(); //returns the number of iterations (policy improvement steps)
    int getNumberOfStates(); //returns the number of states of the general MDP model (-1 for the built-in models)
    int getNumberOfActions(int sidx); //returns the number of actions of a state of the general MDP model (-1 for the built-in models or a state outside the model)
    py::array_t<int64_t> getPolicy(); //returns the entire policy, as a NumPy array (a copy)
    py::array_t<double> getValueVector(); //returns the entire value vector, as a NumPy array (a copy)
    void saveToFile(string fileName, string type, string filetype); //save the policy or value vector to a NumPy .npy file or a comma-separated file (by default result.npy or result.csv when fileName is empty)
    py::dict getConvergence(); //whether the last solve converged, and if not, why

    //------ model check ------

    //checks the general MDP model for malformed input, and returns a description of each
    //kind of problem found (an empty list if none). It is never called by mdp() or solve().
    py::list checkModel(double tolerance=1e-9);

private:

    //METHODS
    void setInitPolicy(py::list initPolicy);
    void setInitValueVector(py::list initValueVector);
    void loadRewardsFromList(py::list rewards);
    void loadTranMatFromLists(py::list tranMatProbs, py::list tranMatColumns);
    void loadTranMatWithZeros(py::list tranMatWithZeros);
    void loadRewardsElementwise(py::list rewardsElementwise);
    void loadTranMatElementwise(py::list tranMatElementwise);
    void loadRewardsFromFile(string rewardsFromFile, char sep, bool header);
    //the same formats as NumPy arrays: mdp() hands over the arrays' data as C-contiguous
    //doubles (int64 for the column indices), and these loaders use no Python
    void loadRewardsFromArray(const double* rewards, int nStates, int nActions);
    void loadRewardsElementwiseArray(const double* rows, int64_t nRows, int64_t rowLength);
    //(the loaders of probabilities read double or float32 arrays, and keep float32 as float32)
    template <typename T> void loadTranMatWithZerosArray(const T* probs, int nStates, int nActions, int nColumns);
    template <typename T> void loadTranMatElementwiseArray(const T* rows, int64_t nRows, int64_t rowLength);
    template <typename T> void loadTranMatFromArrays(const T* probs, const int64_t* columns, int nStates, int nActions, int nEntries);
    void loadTranMatFromFile(string tranMatFromFile, char sep, bool header);
    //the elementwise formats from NumPy .npy files, read in blocks straight into the tables
    void loadRewardsFromNpy(string rewardsFromFile);
    void loadTranMatFromNpy(string tranMatFromFile);
    //the policy or the value vector to a NumPy .npy file (a 1D array) or a comma-separated file
    void savePolicyToNpy(string fileName);
    void saveValueVectorToNpy(string fileName);
    void savePolicyToFile(string fileName, char sep);
    void saveValueVectorToFile(string fileName, char sep);

};

#endif /* MODULEINTERFACE_H */

