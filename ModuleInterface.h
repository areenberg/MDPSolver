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
#include "TBMmodel.h" //Time-based maintenance model
#include "CBMmodel.h" //Condition-based maintenance model
#include "ModifiedPolicyIteration.h" //The solver

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
    
        //only for the TBM/CBM problem
        int components;
        int stages;
        vector<vector<double>> pCompMat; //only for CBM
    } problem;


    //solver settings
    struct Settings{
        string algorithm="mpi";
        double tolerance=1e-3;
        string update = "standard";
        int parIterLim = 100;
        double SORrelaxation = 1.0;
    } settings;


    //results
    struct Results{
        bool evaluated=false;
    } results;

    
    //---------------------------------------
    //  METHODS
    //---------------------------------------

    //------ problem selection ------

    void tbm(double discount,int components,int stages); //select TBM problem
    void cbm(double discount,int components,int stages,py::list pCompMat); //select CBM problem

    //-------------------------------

    void solve(); //solves the problem
    void printPolicy(); //prints the policy

     
private:

};

#endif /* MODULEINTERFACE_H */

