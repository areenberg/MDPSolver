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
        loadRewardsElementwise(rewardsElementwise);
    }else{
        //from file
    }

    //load the transition probabilities
    if (tranMatWithZeros.size()!=0){
        loadTranMatWithZeros(tranMatWithZeros);
    }else if(tranMatElementwise.size()!=0){
        loadTranMatElementwise(tranMatElementwise);
    }else if(tranMatProbs.size()!=0&&tranMatColumns.size()!=0){
        problem.tranMat.assignProbsFromList(tranMatProbs);
        problem.tranMat.assignColumnsFromList(tranMatColumns);
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
        vector<vector<vector<double>>> tempMat = tranMatWithZeros.cast<vector<vector<vector<double>>>>();
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

void ModuleInterface::loadRewardsElementwise(py::list rewardsElementwise){

    py::list innerRew;
    vector<int> nAct;
    int k0,k1;
    double r;
    int numberOfStates=0;
    //determine number of States
    for (int i=0; i<rewardsElementwise.size(); i++){
        innerRew = rewardsElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>();
        if (k0>numberOfStates){
            numberOfStates=k0;
        }
    }       
    numberOfStates++;
    nAct.resize(numberOfStates,0);
    //determine number of actions in each state
    for (int i=0; i<rewardsElementwise.size(); i++){
        innerRew = rewardsElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = innerRew[1].cast<int>();
        if (k1>nAct[k0]){
            nAct[k0]=k1;
        }
    }
    //allocate memory
    problem.rewards.setNumberOfRows(numberOfStates);
    for (int sidx=0; sidx<numberOfStates; sidx++){
        problem.rewards.setNumberOfActions((nAct[sidx]+1),sidx);
    }    
    //assign values
    for (int i=0; i<rewardsElementwise.size(); i++){
        innerRew = rewardsElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = innerRew[1].cast<int>();
        r = innerRew[2].cast<double>();
        problem.rewards.assignReward(r,k0,k1);
    }
}

void ModuleInterface::loadTranMatElementwise(py::list tranMatElementwise){

    py::list innerRew;
    vector<int> nAct;
    vector<vector<int>> nCol;
    int k0,k1,k2,cidx;
    double prob;
    int numberOfStates=0;
    //determine number of States
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = tranMatElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>();
        if (k0>numberOfStates){
            numberOfStates=k0;
        }
    }       
    numberOfStates++;
    nAct.resize(numberOfStates,0);
    nCol.resize(numberOfStates);
    //determine number of actions and columns (next states) in each state
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = tranMatElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = innerRew[1].cast<int>();
        if (k1>nAct[k0]){
            nAct[k0]=k1;
        }
    }
    for (int sidx=0; sidx<numberOfStates; sidx++){
        nCol[sidx].resize((nAct[sidx]+1),0);
    }
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = tranMatElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = innerRew[1].cast<int>();
        k2 = innerRew[2].cast<int>();
        if (k2>nCol[k0][k1]){
            nCol[k0][k1]=k2;
        }
    }
    //allocate memory
    problem.tranMat.setNumberOfRows(numberOfStates);
    for (int sidx=0; sidx<numberOfStates; sidx++){
        problem.tranMat.setNumberOfActions((nAct[sidx]+1),sidx);
        for (int aidx=0; aidx<(nAct[sidx]+1); aidx++){
            problem.tranMat.setNumberOfColumns((nCol[sidx][aidx]+1),sidx,aidx);           
        }
    }    
    //assign values
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = tranMatElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = innerRew[1].cast<int>();
        k2 = innerRew[2].cast<int>(); prob = innerRew[3].cast<double>();
        //find cidx
        cidx=0;
        while (*problem.tranMat.getColumn(k0,k1,cidx)!=-1){
            cidx++;
        }
        problem.tranMat.assignColumn(k2,k0,k1,cidx);
        problem.tranMat.assignProb(prob,k0,k1,cidx);
    }
}