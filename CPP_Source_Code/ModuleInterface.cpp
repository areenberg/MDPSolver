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

#include <fstream>
#include <sstream>

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
    settings.genMDP=true;

    //load the rewards
    if (rewards.size()!=0){
        problem.rewards.assignRewardsFromList(rewards);
    }else if(rewardsElementwise.size()!=0){
        loadRewardsElementwise(rewardsElementwise);
    }else{
        loadRewardsFromFile(rewardsFromFile,',',true);
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
        loadTranMatFromFile(tranMatFromFile,',',true);
    }

}

void ModuleInterface::tbm(double discount,
    int components,
    int stages,
    double replacementCost,
    double setupCost,
    double unexpectedFailureCost,
    double expiredNotFixedCost,
    double failureProb,
    double failureProbMin,
    double failureProbHat){
    //selects the TBM problem
    problem.problemType="tbm";
    settings.genMDP=false;
    problem.discount=discount;
    problem.components=components;
    problem.stages=stages;
    problem.replacementCost=replacementCost;
    problem.setupCost=setupCost;
    problem.unexpectedFailureCost=unexpectedFailureCost;
    problem.expiredNotFixedCost=expiredNotFixedCost;
    problem.failureProb=failureProb;
    problem.failureProbMin=failureProbMin;
    problem.failureProbHat=failureProbHat;
    //cout << "Selected time-based maintenance problem with " << problem.components <<
    // " components and " << problem.stages << " stages." << endl;
}

void ModuleInterface::cbm(double discount,
    int components,
    int stages,
    py::list pCompMat,
    double preventiveCost,
    double correctiveCost,
    double setupCost,
    double failurePenalty,
    int kOfN){
    //selects the CBM problem
    problem.problemType="cbm";
    settings.genMDP=false;
    problem.discount=discount;
    problem.components=components;
    problem.stages=stages;
    problem.pCompMat=pCompMat.cast<vector<vector<double>>>();
    problem.preventiveCost=preventiveCost;
    problem.correctiveCost=correctiveCost;
    problem.setupCost=setupCost;
    problem.failurePenalty=failurePenalty;
    problem.kOfN=kOfN;
    //cout << "Selected condition-based maintenance problem with " << problem.components <<
    // " components and " << problem.stages << " stages." << endl;
}


void ModuleInterface::solve(string algorithm,
                            double tolerance,
                            string update,
                            int parIterLim,
                            double SORrelaxation,
                            py::list initPolicy,
                            py::list initValueVector,
                            bool verbose,
                            bool postProcessing,
                            bool makeFinalCheck,
                            bool parallel){

    //store solver settings
    settings.algorithm=algorithm;
    settings.tolerance=tolerance;
    settings.update=update;
    settings.parIterLim=parIterLim;
    settings.SORrelaxation=SORrelaxation;
    settings.verbose=verbose;
    settings.postProcessing=postProcessing;
    settings.makeFinalCheck=makeFinalCheck;
    settings.parallel=parallel;
    setInitPolicy(initPolicy);
    setInitValueVector(initValueVector);

    //create and setup solver object
    ModifiedPolicyIteration solver(settings.tolerance, settings.algorithm, 
    settings.update, settings.parIterLim, settings.SORrelaxation, settings.verbose,
    settings.postProcessing, settings.makeFinalCheck, settings.parallel, settings.genMDP);

    //create model object
    if (problem.problemType.compare("mdp")==0){
        GeneralMDPmodel mdl(&problem.rewards,&problem.tranMat,problem.discount); //General MDP model
        solver.solve(&mdl,&problem.policy,&problem.valueVector);
    }else if (problem.problemType.compare("tbm")==0){
        TBMmodel mdl(problem.discount, //Time-based maintenance model
        problem.components,
        problem.stages,
        problem.replacementCost,
        problem.setupCost,
        problem.unexpectedFailureCost,
        problem.expiredNotFixedCost,
        problem.failureProb,
        problem.failureProbMin,
        problem.failureProbHat);
        solver.solve(&mdl,&problem.policy,&problem.valueVector);
    }else if(problem.problemType.compare("cbm")==0){
        CBMmodel mdl(problem.discount, //Condition-based maintenance model
        problem.components,
        problem.stages,
        problem.pCompMat,
        problem.preventiveCost,
        problem.correctiveCost,
        problem.setupCost,
        problem.failurePenalty,
        problem.kOfN);
        solver.solve(&mdl,&problem.policy,&problem.valueVector);
    }

    //save duration (runtime) in milliseconds
    results.duration=solver.duration;
        
}

void ModuleInterface::setInitPolicy(py::list initPolicy){
    if (initPolicy.size()>0){
        //cout << "Initializing with policy:" << endl;
        problem.policy.setSize(initPolicy.size());
        for (int sidx=0; sidx<initPolicy.size(); sidx++){
            int a = initPolicy[sidx].cast<int>();
            //cout << a << endl;
            problem.policy.assignPolicy(sidx,a);    
        }
    }
}

void ModuleInterface::setInitValueVector(py::list initValueVector){
    if (initValueVector.size()>0){
        //cout << "Initializing with values:" << endl;
        problem.valueVector.setSize(initValueVector.size());
        for (int sidx=0; sidx<initValueVector.size(); sidx++){
            double v = initValueVector[sidx].cast<double>();
            //cout << v << endl;
            problem.valueVector.assignValue(sidx,v);    
        }
    }    
}

py::list ModuleInterface::getPolicy(){
    return(py::cast(problem.policy.policy));
}

py::list ModuleInterface::getValueVector(){
    return(py::cast(problem.valueVector.valueVector));
}

int ModuleInterface::getAction(int sidx){
    if (sidx>=0 && sidx<problem.policy.policy.size()){
        return(problem.policy.policy[sidx]);
    }else{
        return(-1);
    }
}

double ModuleInterface::getValue(int sidx){
    if (sidx>=0 && sidx<problem.valueVector.valueVector.size()){
        return(problem.valueVector.valueVector[sidx]);
    }else{
        return(0.0);
    }
}

void ModuleInterface::printPolicy(){
    for (int sidx=0; sidx<problem.policy.policy.size(); sidx++){
        cout << sidx << ": " << problem.policy.policy[sidx] << endl; 
    }
}

void ModuleInterface::printValueVector(){
    for (int sidx=0; sidx<problem.valueVector.valueVector.size(); sidx++){
        cout << sidx << ": " << problem.valueVector.valueVector[sidx] << endl; 
    }
}

void ModuleInterface::saveToFile(string fileName, string type){
    if (type.compare("policy")==0 || type.compare("p")==0){
        savePolicyToFile(fileName,',');    
    }else if(type.compare("values")==0 || type.compare("v")==0){
        saveValueVectorToFile(fileName,',');
    }
}

void ModuleInterface::savePolicyToFile(string fileName, char sep){
    ofstream outFile(fileName);
    if (!outFile.is_open()) {
        cerr << "Failed to open file: " << fileName << endl;
        return;
    }
    outFile << "State_Index" << sep << "Action_Index" << endl;
    for (int sidx = 0; sidx < problem.policy.policy.size(); sidx++) {
        outFile << sidx << sep << problem.policy.policy[sidx] << endl;
    }
    outFile.close();
}

void ModuleInterface::saveValueVectorToFile(string fileName, char sep){
    ofstream outFile(fileName);
    if (!outFile.is_open()) {
        cerr << "Failed to open file: " << fileName << endl;
        return;
    }
    outFile << "State_Index" << sep << "Value" << endl;
    for (int sidx = 0; sidx < problem.valueVector.valueVector.size(); sidx++) {
        outFile << sidx << sep << problem.valueVector.valueVector[sidx] << endl;
    }
    outFile.close();
}

double ModuleInterface::getRuntime(){
    return(results.duration);
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

void ModuleInterface::loadRewardsFromFile(string rewardsFromFile, char sep, bool header){

    string line,cell;
    vector<int> nAct;
    int i,k0,k1;
    double r;
    int numberOfStates=0;

    //determine number of states
    ifstream file(rewardsFromFile);
    if (!file.is_open()) {
        cerr << "Error: Unable to open " << rewardsFromFile << endl;
        return;
    }
    i=0;
    while (getline(file,line)){
        if (!header||i>0){
            stringstream lineStream(line);
            vector<string> innerRew;
            while (getline(lineStream,cell,sep)) {
                innerRew.push_back(cell);
            }
            k0 = stoi(innerRew[0]);
            if (k0>numberOfStates){
                numberOfStates=k0;
            }
        }
        i++;
    }
    numberOfStates++;    
    nAct.resize(numberOfStates,0);
    
    //reset
    file.clear();
    file.seekg(0,ios::beg);

    //determine number of actions in each state
    i=0;
    while (getline(file,line)){
        if (!header||i>0){
            stringstream lineStream(line);
            vector<string> innerRew;
            while (getline(lineStream,cell,sep)) {
                innerRew.push_back(cell); 
                
            }
            k0 = stoi(innerRew[0]); k1 = stoi(innerRew[1]);
            if (k1>nAct[k0]){
                nAct[k0]=k1;
            }    
        }
        i++;
    }
    
    //reset
    file.clear();
    file.seekg(0,ios::beg);

    //allocate memory
    problem.rewards.setNumberOfRows(numberOfStates);
    for (int sidx=0; sidx<numberOfStates; sidx++){
        problem.rewards.setNumberOfActions((nAct[sidx]+1),sidx);
    }

    //assign values
    i=0;
    while (getline(file,line)){
        if (!header||i>0){
            stringstream lineStream(line);
            vector<string> innerRew;
            while (getline(lineStream,cell,sep)) {
                innerRew.push_back(cell);
            }
            k0 = stoi(innerRew[0]); k1 = stoi(innerRew[1]);
            r = stod(innerRew[2]);
            problem.rewards.assignReward(r,k0,k1);    
        }
        i++;
    }
    file.close();
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
        while (problem.tranMat.getColumn(k0,k1,cidx)!=-1){
            cidx++;
        }
        problem.tranMat.assignColumn(k2,k0,k1,cidx);
        problem.tranMat.assignProb(prob,k0,k1,cidx);
    }
}


void ModuleInterface::loadTranMatFromFile(string tranMatFromFile, char sep, bool header){
    
    string line,cell;
    vector<int> nAct;
    vector<vector<int>> nCol;
    int i,k0,k1,k2,cidx;
    double prob;
    int numberOfStates=0;

   //determine number of States
   ifstream file(tranMatFromFile);
    if (!file.is_open()) {
        cerr << "Error: Unable to open " << tranMatFromFile << endl;
        return;
    }
    i=0;
    while (getline(file,line)){
        if (!header||i>0){
            stringstream lineStream(line);
            vector<string> innerRew;
            while (getline(lineStream,cell,sep)) {
                innerRew.push_back(cell);
            }
            k0 = stoi(innerRew[0]);
            if (k0>numberOfStates){
                numberOfStates=k0;
            }
        }
        i++;
    }
    numberOfStates++;
    nAct.resize(numberOfStates,0);
    nCol.resize(numberOfStates);

    //reset
    file.clear();
    file.seekg(0,ios::beg);

    //determine number of actions and columns (next states) in each state
    i=0;
    while (getline(file,line)){
        if (!header||i>0){
            stringstream lineStream(line);
            vector<string> innerRew;
            while (getline(lineStream,cell,sep)) {
                innerRew.push_back(cell); 
                
            }
            k0 = stoi(innerRew[0]); k1 = stoi(innerRew[1]);
            if (k1>nAct[k0]){
                nAct[k0]=k1;
            }    
        }
        i++;
    }
    for (int sidx=0; sidx<numberOfStates; sidx++){
        nCol[sidx].resize((nAct[sidx]+1),0);
    }

    //reset
    file.clear();
    file.seekg(0,ios::beg);

    i=0;
    while (getline(file,line)){
        if (!header||i>0){
            stringstream lineStream(line);
            vector<string> innerRew;
            while (getline(lineStream,cell,sep)) {
                innerRew.push_back(cell); 
                
            }
            k0 = stoi(innerRew[0]); k1 = stoi(innerRew[1]);
            k2 = stoi(innerRew[2]);
            if (k2>nCol[k0][k1]){
                nCol[k0][k1]=k2;
            }    
        }
        i++;
    }

    //reset
    file.clear();
    file.seekg(0,ios::beg);

    //allocate memory
    problem.tranMat.setNumberOfRows(numberOfStates);
    for (int sidx=0; sidx<numberOfStates; sidx++){
        problem.tranMat.setNumberOfActions((nAct[sidx]+1),sidx);
        for (int aidx=0; aidx<(nAct[sidx]+1); aidx++){
            problem.tranMat.setNumberOfColumns((nCol[sidx][aidx]+1),sidx,aidx);           
        }
    }

    //assign values
    i=0;
    while (getline(file,line)){
        if (!header||i>0){
            stringstream lineStream(line);
            vector<string> innerRew;
            while (getline(lineStream,cell,sep)) {
                innerRew.push_back(cell);
            }
            k0 = stoi(innerRew[0]); k1 = stoi(innerRew[1]);
            k2 = stoi(innerRew[2]); prob = stod(innerRew[3]);
            //find cidx
            cidx=0;
            while (problem.tranMat.getColumn(k0,k1,cidx)!=-1){
                cidx++;
            }
            problem.tranMat.assignColumn(k2,k0,k1,cidx);
            problem.tranMat.assignProb(prob,k0,k1,cidx);
        }
        i++;
    }
    file.close();
}