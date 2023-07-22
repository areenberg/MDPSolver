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

void ModuleInterface::mdp(double discount,
    py::list rewards, 
    py::list rewardsElementwise,
    string rewardsFromFile,
    py::list tranMatWithZeros,
    py::list tranMatElementwise,
    py::list tranMatProbs,
    py::list tranMatColumns, 
    string tranMatFromFile){
    //select the general MDP problem
    problem.problemType="mdp";
    problem.discount=discount;

    //load the rewards
    if (rewards.size()!=0){
        problem.rewards.assignRewardsFromList(rewards);
    }else if(rewardsElementwise.size()!=0){

    }else{
        //from file
    }

    //load the transition probabilities
    if (tranMatWithZeros.size()!=0){
        loadTranMatWithZeros(tranMatWithZeros);
    }else if(tranMatElementwise.size()!=0){

    }else if(tranMatProbs.size()!=0&&tranMatColumns.size()!=0){

    }else{
        //from file
    }

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


void ModuleInterface::solve(string algorithm, double tolerance, string update, int parIterLim, double SORrelaxation){

    //store solver settings
    settings.algorithm=algorithm;
    settings.tolerance=tolerance;
    settings.update=update;
    settings.parIterLim=parIterLim;
    settings.SORrelaxation=SORrelaxation;

    //create and setup solver object
    ModifiedPolicyIteration solver(settings.tolerance, settings.algorithm, 
    settings.update, settings.parIterLim, settings.SORrelaxation);

    //create model object
    if (problem.problemType.compare("mdp")==0){
        GeneralMDPmodel mdl(&problem.rewards,&problem.tranMat,problem.discount); //General MDP model
        solver.solve(&mdl,&problem.policy,&problem.valueVector);
    }else if (problem.problemType.compare("tbm")==0){
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

void ModuleInterface::loadTranMatWithZeros(py::list tranMatWithZeros){
        vector<vector<vector<double>>> tempMat = tranMatWithZeros.cast< vector<vector<vector<double>>>  >();
        int cidx;
        problem.tranMat.setNumberOfRows(tempMat.size());
        for (int sidx=0; sidx<tempMat.size(); sidx++){
            problem.tranMat.setNumberOfActions(tempMat[sidx].size(),sidx);
            for (int aidx=0; aidx<tempMat[sidx].size(); aidx++){
                cidx=0;
                for (int jidx=0; jidx<tempMat[sidx][aidx].size(); jidx++){
                    if (tempMat[sidx][aidx][jidx]>0.0){
                        cidx++;
                    }
                }
                problem.tranMat.setNumberOfColumns(cidx,sidx,aidx);
                cidx=0;
                for (int jidx=0; jidx<tempMat[sidx][aidx].size(); jidx++){
                    if (tempMat[sidx][aidx][jidx]>0.0){
                        problem.tranMat.assignColumn(jidx,sidx,aidx,cidx);
                        problem.tranMat.assignProb(tempMat[sidx][aidx][jidx],sidx,aidx,cidx);
                        cidx++;
                    }
                }
            } 
        }
}