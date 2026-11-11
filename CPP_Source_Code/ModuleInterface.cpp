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

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <cmath>

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
                            string criterion,
                            int parIterLim,
                            double SORrelaxation,
                            py::list initPolicy,
                            py::list initValueVector,
                            bool verbose,
                            bool postProcessing,
                            bool makeFinalCheck,
                            bool parallel){

    if (problem.problemType.empty()){
        throw runtime_error("No model has been defined. Call mdp() before solve().");
    }

    //store solver settings
    settings.algorithm=algorithm;
    settings.tolerance=tolerance;
    settings.update=update;
    settings.criterion=criterion;
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
    settings.update, settings.criterion, settings.parIterLim, settings.SORrelaxation, settings.verbose,
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
    //save the number of iterations (policy improvement steps)
    results.iterations=solver.iter;
    //save whether the solver converged, and if not, why
    results.converged=solver.converged;
    results.iterationLimitReached=solver.iterationLimitReached;
    results.evaluationLimitReached=solver.evaluationLimitReached;
    results.valuesOutOfRange=solver.valuesOutOfRange;
    results.diverged=solver.diverged;
    results.outOfRangeState=solver.outOfRangeState;
    results.outOfRangeValue=solver.outOfRangeValue;
    results.iterationLimit=solver.getIterationLimit();
        
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
        problem.policy.isSet=true;
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
        problem.valueVector.isSet=true;
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
        //pybind11 turns std::out_of_range into IndexError
        throw out_of_range("getAction(): the state index " + to_string(sidx) + " is outside the model, whose states are 0 to " + to_string((int)problem.policy.policy.size()-1) + ".");
    }
}

double ModuleInterface::getValue(int sidx){
    if (sidx>=0 && sidx<problem.valueVector.valueVector.size()){
        return(problem.valueVector.valueVector[sidx]);
    }else{
        throw out_of_range("getValue(): the state index " + to_string(sidx) + " is outside the model, whose states are 0 to " + to_string((int)problem.valueVector.valueVector.size()-1) + ".");
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
    }else if(type.compare("value")==0 || type.compare("values")==0 || type.compare("v")==0){
        saveValueVectorToFile(fileName,',');
    }else{
        throw invalid_argument("saveToFile(): the type '" + type + "' is not recognized. Select either 'policy' or 'value'.");
    }
}

void ModuleInterface::savePolicyToFile(string fileName, char sep){
    ofstream outFile(fileName);
    if (!outFile.is_open()) {
        throw runtime_error("saveToFile(): failed to open the file " + fileName + " for writing.");
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
        throw runtime_error("saveToFile(): failed to open the file " + fileName + " for writing.");
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

int ModuleInterface::getIterations(){
    return(results.iterations);
}

py::dict ModuleInterface::getConvergence(){
    py::dict convergence;
    convergence["converged"]=results.converged;
    convergence["iterationLimitReached"]=results.iterationLimitReached;
    convergence["evaluationLimitReached"]=results.evaluationLimitReached;
    convergence["valuesOutOfRange"]=results.valuesOutOfRange;
    convergence["diverged"]=results.diverged;
    convergence["outOfRangeState"]=results.outOfRangeState;
    convergence["outOfRangeValue"]=results.outOfRangeValue;
    convergence["iterationLimit"]=results.iterationLimit;
    return(convergence);
}

int ModuleInterface::getNumberOfStates(){
    //the number of states of the general MDP model
    if (problem.problemType.compare("mdp")==0){
        return(problem.tranMat.numberOfRows());
    }else{
        return(-1);
    }
}

int ModuleInterface::getNumberOfActions(int sidx){
    //the number of actions of a state
    if (problem.problemType.compare("mdp")==0 && sidx>=0 && sidx<problem.tranMat.numberOfRows()){
        return(problem.tranMat.numberOfActions(sidx));
    }else{
        return(-1);
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

//---------------------------------------
//  LOADING: crash guards
//---------------------------------------
//The loaders check only what would otherwise make them crash: a row with too few
//entries, an index that is not a non-negative integer, and a number that cannot be
//read. Blank lines in files are skipped. Everything else is left to checkModel(),
//which the user calls when needed. (pybind11 turns std::invalid_argument into
//ValueError.)

//a row of an elementwise list, with at least n entries
static py::list listRow(const py::list& rows, int i, size_t n, const string& name, const string& columns){
    py::list row = rows[i].cast<py::list>();
    if (row.size() < n){
        throw invalid_argument(name + "[" + to_string(i) + "] has " + to_string(row.size()) + " entries, where " + to_string(n) + " are needed (" + columns + ").");
    }
    return row;
}

//an index in a row of an elementwise list, which must not be negative
static int listIndex(const py::list& row, int k, int i, const string& name, const string& what){
    int index = row[k].cast<int>();
    if (index < 0){
        throw invalid_argument(name + "[" + to_string(i) + "] has the negative " + what + " index " + to_string(index) + ".");
    }
    return index;
}

//splits a line of a file into its fields; false for a blank line, an error for too few fields
static bool fileRow(const string& line, char sep, size_t n, const string& fileName, int lineNumber, const string& columns, vector<string>& fields){
    fields.clear();
    if (line.find_first_not_of(" \t\r") == string::npos){
        return false;
    }
    stringstream lineStream(line);
    string cell;
    while (getline(lineStream,cell,sep)){
        fields.push_back(cell);
    }
    if (fields.size() < n){
        throw invalid_argument("Line " + to_string(lineNumber) + " of " + fileName + " has " + to_string(fields.size()) + " fields, where " + to_string(n) + " are needed (" + columns + ").");
    }
    return true;
}

//an index in a field of a file, which must be a non-negative integer
static int fileIndex(const string& field, const string& fileName, int lineNumber, const string& what){
    int index;
    try{
        index = stoi(field);
    }catch(const exception&){
        throw invalid_argument("Line " + to_string(lineNumber) + " of " + fileName + " has the " + what + " index '" + field + "', which is not an integer.");
    }
    if (index < 0){
        throw invalid_argument("Line " + to_string(lineNumber) + " of " + fileName + " has the negative " + what + " index " + to_string(index) + ".");
    }
    return index;
}

//a number in a field of a file
static double fileNumber(const string& field, const string& fileName, int lineNumber, const string& what){
    try{
        return stod(field);
    }catch(const exception&){
        throw invalid_argument("Line " + to_string(lineNumber) + " of " + fileName + " has the " + what + " '" + field + "', which is not a number.");
    }
}

void ModuleInterface::loadRewardsElementwise(py::list rewardsElementwise){

    const string name = "rewardsElementwise";
    const string columns = "state, action, reward";
    py::list innerRew;
    vector<int> nAct;
    int k0,k1;
    double r;
    int numberOfStates=0;
    //determine number of States
    for (int i=0; i<rewardsElementwise.size(); i++){
        innerRew = listRow(rewardsElementwise,i,3,name,columns);
        k0 = listIndex(innerRew,0,i,name,"state");
        if (k0>numberOfStates){
            numberOfStates=k0;
        }
    }       
    numberOfStates++;
    nAct.resize(numberOfStates,0);
    //determine number of actions in each state
    for (int i=0; i<rewardsElementwise.size(); i++){
        innerRew = rewardsElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = listIndex(innerRew,1,i,name,"action");
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

    const string columns = "state, action, reward";
    string line;
    vector<string> fields;
    vector<int> nAct;
    int i,k0,k1;
    double r;
    int numberOfStates=0;

    //determine number of states
    ifstream file(rewardsFromFile);
    if (!file.is_open()) {
        throw runtime_error("Unable to open the rewards file " + rewardsFromFile + ".");
    }
    i=0;
    while (getline(file,line)){
        if ((!header||i>0) && fileRow(line,sep,3,rewardsFromFile,i+1,columns,fields)){
            k0 = fileIndex(fields[0],rewardsFromFile,i+1,"state");
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
        if ((!header||i>0) && fileRow(line,sep,3,rewardsFromFile,i+1,columns,fields)){
            k0 = stoi(fields[0]); k1 = fileIndex(fields[1],rewardsFromFile,i+1,"action");
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
        if ((!header||i>0) && fileRow(line,sep,3,rewardsFromFile,i+1,columns,fields)){
            k0 = stoi(fields[0]); k1 = stoi(fields[1]);
            r = fileNumber(fields[2],rewardsFromFile,i+1,"reward");
            problem.rewards.assignReward(r,k0,k1);    
        }
        i++;
    }
    file.close();
}

void ModuleInterface::loadTranMatElementwise(py::list tranMatElementwise){

    const string name = "tranMatElementwise";
    const string columns = "state, action, next state, probability";
    py::list innerRew;
    vector<int> nAct;
    vector<vector<int>> nCol;
    int k0,k1,k2,cidx;
    double prob;
    int numberOfStates=0;
    //determine number of States
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = listRow(tranMatElementwise,i,4,name,columns);
        k0 = listIndex(innerRew,0,i,name,"state");
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
        k0 = innerRew[0].cast<int>(); k1 = listIndex(innerRew,1,i,name,"action");
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
        nCol[k0][k1]++;
    }
    //allocate memory
    problem.tranMat.setNumberOfRows(numberOfStates);
    for (int sidx=0; sidx<numberOfStates; sidx++){
        problem.tranMat.setNumberOfActions((nAct[sidx]+1),sidx);
        for (int aidx=0; aidx<(nAct[sidx]+1); aidx++){
            problem.tranMat.setNumberOfColumns(nCol[sidx][aidx],sidx,aidx);
        }
    }
    //assign values
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = tranMatElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = innerRew[1].cast<int>();
        k2 = listIndex(innerRew,2,i,name,"next-state"); prob = innerRew[3].cast<double>();
        //find cidx (the next free place, marked -1, which a negative k2 would leave free)
        cidx=0;
        while (problem.tranMat.getColumn(k0,k1,cidx)!=-1){
            cidx++;
        }
        problem.tranMat.assignColumn(k2,k0,k1,cidx);
        problem.tranMat.assignProb(prob,k0,k1,cidx);
    }
}


void ModuleInterface::loadTranMatFromFile(string tranMatFromFile, char sep, bool header){
    
    const string columns = "state, action, next state, probability";
    string line;
    vector<string> fields;
    vector<int> nAct;
    vector<vector<int>> nCol;
    int i,k0,k1,k2,cidx;
    double prob;
    int numberOfStates=0;

   //determine number of States
   ifstream file(tranMatFromFile);
    if (!file.is_open()) {
        throw runtime_error("Unable to open the transition probabilities file " + tranMatFromFile + ".");
    }
    i=0;
    while (getline(file,line)){
        if ((!header||i>0) && fileRow(line,sep,4,tranMatFromFile,i+1,columns,fields)){
            k0 = fileIndex(fields[0],tranMatFromFile,i+1,"state");
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
        if ((!header||i>0) && fileRow(line,sep,4,tranMatFromFile,i+1,columns,fields)){
            k0 = stoi(fields[0]); k1 = fileIndex(fields[1],tranMatFromFile,i+1,"action");
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
        if ((!header||i>0) && fileRow(line,sep,4,tranMatFromFile,i+1,columns,fields)){
            k0 = stoi(fields[0]); k1 = stoi(fields[1]);
            nCol[k0][k1]++;
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
            problem.tranMat.setNumberOfColumns(nCol[sidx][aidx],sidx,aidx);
        }
    }

    //assign values
    i=0;
    while (getline(file,line)){
        if ((!header||i>0) && fileRow(line,sep,4,tranMatFromFile,i+1,columns,fields)){
            k0 = stoi(fields[0]); k1 = stoi(fields[1]);
            k2 = fileIndex(fields[2],tranMatFromFile,i+1,"next-state");
            prob = fileNumber(fields[3],tranMatFromFile,i+1,"probability");
            //find cidx (the next free place, marked -1, which a negative k2 would leave free)
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


//---------------------------------------
//  MODEL CHECK
//---------------------------------------
//Called only by the user (model.checkModel() in Python), never by mdp() or solve(),
//so that solving a model costs no time on checks. It reads the loaded model with
//bounds checks of its own, so that it does not crash on the models it is meant to
//find, and reports each kind of problem once, with a count and the first examples.

namespace {

//one kind of problem: how many were found, and the first few
struct ProblemKind{
    long long count=0;
    vector<string> examples;
    void add(const string& example){
        count++;
        if (examples.size() < 3){
            examples.push_back(example);
        }
    }
    //"<count> <what>: <examples>", or ", e.g. <examples>" when there are more than shown
    void report(vector<string>& problems, const string& singular, const string& plural) const{
        if (count == 0){
            return;
        }
        string message = to_string(count) + " " + (count == 1 ? singular : plural);
        message += (count > (long long)examples.size()) ? ", e.g. " : ": ";
        for (size_t k=0; k<examples.size(); k++){
            message += (k > 0 ? "; " : "") + examples[k];
        }
        problems.push_back(message + ".");
    }
};

//a number with ten significant digits, so that a sum of 0.9999999 does not show as 1
string number(double x){
    ostringstream text;
    text.precision(10);
    text << x;
    return text.str();
}

//"1 state", "2 states"
string counted(long long n, const string& singular, const string& plural){
    return to_string(n) + " " + (n == 1 ? singular : plural);
}

string stateAction(int sidx, int aidx){
    return "state " + to_string(sidx) + ", action " + to_string(aidx);
}

}

py::list ModuleInterface::checkModel(double tolerance){

    if (problem.problemType.compare("mdp")!=0){
        throw runtime_error("No model has been defined with mdp(), so there is no model to check.");
    }
    if (!(tolerance >= 0.0)){
        throw invalid_argument("checkModel(): the tolerance needs to be zero or positive.");
    }

    vector<string> problems;
    TransitionMatrix& tranMat = problem.tranMat;
    Rewards& rewards = problem.rewards;
    int nStates = tranMat.numberOfRows();
    int nRewardStates = rewards.numberOfRows();
    int nColumnStates = tranMat.numberOfColumnRows();

    //the numbers of states
    if (nStates == 0){
        problems.push_back("The model has no states.");
    }
    if (nRewardStates != nStates){
        problems.push_back("The rewards cover " + counted(nRewardStates,"state","states") + ", but the transition probabilities cover " + to_string(nStates) + ".");
    }
    if (nColumnStates != nStates){
        problems.push_back("tranMatProbs covers " + counted(nStates,"state","states") + ", but tranMatColumns covers " + to_string(nColumnStates) + ".");
    }

    ProblemKind noActions, actionMismatch, shapeMismatch, noTransitions, nextStateOutside, badProbability, badSum, badReward;

    for (int sidx=0; sidx<nStates; sidx++){
        int nActions = tranMat.numberOfActions(sidx);
        if (nActions == 0){
            noActions.add("state " + to_string(sidx));
        }
        if (sidx < nRewardStates && rewards.numberOfActions(sidx) != nActions){
            actionMismatch.add("state " + to_string(sidx) + " (" + to_string(rewards.numberOfActions(sidx)) + " in the rewards, " + to_string(nActions) + " in the transition probabilities)");
        }
        if (sidx >= nColumnStates){
            continue;
        }
        int nColumnActions = tranMat.numberOfColumnActions(sidx);
        if (nColumnActions != nActions){
            shapeMismatch.add("state " + to_string(sidx) + " (" + counted(nActions,"action","actions") + " in tranMatProbs, " + to_string(nColumnActions) + " in tranMatColumns)");
        }

        for (int aidx=0; aidx<min(nActions,nColumnActions); aidx++){
            int nProbs = tranMat.numberOfProbabilities(sidx,aidx);
            int nColumns = tranMat.numberOfColumns(sidx,aidx);
            if (nProbs != nColumns){
                shapeMismatch.add(stateAction(sidx,aidx) + " (" + counted(nProbs,"entry","entries") + " in tranMatProbs, " + to_string(nColumns) + " in tranMatColumns)");
            }
            int nJumps = min(nProbs,nColumns);
            if (nJumps == 0){
                noTransitions.add(stateAction(sidx,aidx));
                continue;
            }
            double sum = 0.0;
            for (int cidx=0; cidx<nJumps; cidx++){
                int jidx = tranMat.getColumn(sidx,aidx,cidx);
                double prob = tranMat.getProb(sidx,aidx,cidx);
                if (jidx < 0 || jidx >= nStates){
                    nextStateOutside.add(stateAction(sidx,aidx) + ", next state " + to_string(jidx));
                }
                if (!std::isfinite(prob) || prob < 0.0 || prob > 1.0){
                    badProbability.add(stateAction(sidx,aidx) + ", next state " + to_string(jidx) + " (" + number(prob) + ")");
                }
                sum += prob;
            }
            if (std::isfinite(sum) && fabs(sum - 1.0) > tolerance){
                badSum.add(stateAction(sidx,aidx) + " (sum " + number(sum) + ")");
            }
        }
    }

    for (int sidx=0; sidx<nRewardStates; sidx++){
        for (int aidx=0; aidx<rewards.numberOfActions(sidx); aidx++){
            double reward = rewards.getReward(sidx,aidx);
            if (!std::isfinite(reward)){
                badReward.add(stateAction(sidx,aidx) + (std::isnan(reward) ? " (missing or NaN)" : " (" + number(reward) + ")"));
            }
        }
    }

    noActions.report(problems, "state has no actions", "states have no actions");
    actionMismatch.report(problems, "state has a different number of actions in the rewards and in the transition probabilities",
        "states have a different number of actions in the rewards and in the transition probabilities");
    shapeMismatch.report(problems, "place where tranMatProbs and tranMatColumns differ in shape",
        "places where tranMatProbs and tranMatColumns differ in shape");
    noTransitions.report(problems, "state-action pair has no transition probabilities", "state-action pairs have no transition probabilities");
    nextStateOutside.report(problems, "transition goes to a next state outside the model, whose states are 0 to " + to_string(nStates-1),
        "transitions go to a next state outside the model, whose states are 0 to " + to_string(nStates-1));
    badProbability.report(problems, "transition probability is not a number between 0 and 1", "transition probabilities are not numbers between 0 and 1");
    badSum.report(problems, "state-action pair has transition probabilities that do not sum to 1 (within " + number(tolerance) + ")",
        "state-action pairs have transition probabilities that do not sum to 1 (within " + number(tolerance) + ")");
    badReward.report(problems, "state-action pair has a reward that is missing or not a finite number",
        "state-action pairs have a reward that is missing or not a finite number");

    return py::cast(problems);
}
