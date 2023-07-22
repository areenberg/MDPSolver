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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
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
        vector<vector<double>> pCompMat; //only for CBM
    } problem;


    //solver settings
    struct Settings{
        string algorithm;
        double tolerance;
        string update;
        int parIterLim;
        double SORrelaxation;
    } settings;


    //results
    struct Results{
        bool evaluated=false;
    } results;

    
    //---------------------------------------
    //  METHODS
    //---------------------------------------

    //------ problem selection ------

    //the general MDP problem
    void mdp(double discount,
    py::list rewards, //option1: complete reward list 
    py::list rewardsElementwise, //option2: rewards where each row is an element and columns specify sidx,aidx,reward
    string rewardsFromFile, //option3: rewards are loaded from a file
    py::list tranMatWithZeros, //option1: complete transition mat incl. zeros
    py::list tranMatElementwise, //option2: tran mat where each row is a non-zero element and columns specify sidx,aidx,jidx,prob
    py::list tranMatProbs, //option3a: transition mat non-zero probabilities
    py::list tranMatColumns, //option3b: transition mat column indices 
    string tranMatFromFile); //option4: transition mat is loaded from a file

    //pre-defined MDP problems  
    void tbm(double discount,int components,int stages); //select TBM problem
    void cbm(double discount,int components,int stages,py::list pCompMat); //select CBM problem

    //-------------------------------

    void solve(string algorithm, double tolerance, string update, int parIterLim, double SORrelaxation); //solves the problem
    void printPolicy(); //prints the policy

     
private:

    //METHODS
    void loadTranMatWithZeros(py::list tranMatWithZeros);

};

#endif /* MODULEINTERFACE_H */

