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

#include "ModuleInterface.h"

ModuleInterface::ModuleInterface() {
}

ModuleInterface::ModuleInterface(const ModuleInterface& orig) {
}

ModuleInterface::~ModuleInterface() {
}


void ModuleInterface::tbm(double discount,int components,int stages){
    //selects the TBM problem
    problem.problemType="tbm";
    problem.discount=discount;
    problem.components=components;
    problem.stages=stages;
    cout << "Selected time-based maintenance problem with " << problem.components <<
     " components and " << problem.stages << " stages." << endl;
}

void ModuleInterface::cbm(double discount,int components,int stages,
py::list pCompMat){
    //selects the CBM problem
    problem.problemType="cbm";
    problem.discount=discount;
    problem.components=components;
    problem.stages=stages;
    problem.pCompMat=pCompMat.cast<vector<vector<double>>>();
    cout << "Selected condition-based maintenance problem with " << problem.components <<
     " components and " << problem.stages << " stages." << endl;
}


void ModuleInterface::solve(){

    //create and setup solver object
    ModifiedPolicyIteration solver(settings.tolerance, settings.algorithm, 
    settings.update, settings.parIterLim, settings.SORrelaxation);

    //create model object
    if (problem.problemType.compare("tbm")==0){
        TBMmodel mdl(problem.components,problem.stages,problem.discount); //Time-based maintenance model
        solver.solve(&mdl,&problem.policy,&problem.valueVector);
    }else if(problem.problemType.compare("cbm")==0){
        CBMmodel mdl(problem.components,problem.stages,problem.discount,problem.pCompMat); //Condition-based maintenance model
        solver.solve(&mdl,&problem.policy,&problem.valueVector);
    }
        
}

void ModuleInterface::printPolicy(){
    for (int sidx=0; sidx<problem.policy.policy.size(); sidx++){
        cout << sidx << ": " << problem.policy.policy[sidx] << endl; 
    }
}
