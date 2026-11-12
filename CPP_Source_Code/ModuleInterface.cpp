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
#include <climits>

#include "ModuleInterface.h"

ModuleInterface::ModuleInterface() {
}

ModuleInterface::ModuleInterface(const ModuleInterface& orig) {
}

ModuleInterface::~ModuleInterface() {
}

//---------------------------------------
//  mdp(): lists or NumPy arrays
//---------------------------------------
//Each format but the files can be given as a list or as a NumPy array. A list goes to the
//list loaders below, and never makes Python import NumPy. Anything else is taken as an array
//(model.py has already checked its shape and converted it to C-contiguous float64, or int64
//for the column indices), and only the few lines here touch it through pybind11: the array
//loaders get plain pointers and sizes.

namespace {

//whether a format was given: a non-empty list or array
bool given(const py::object& argument){
    return !argument.is_none() && py::len(argument) > 0;
}

//an argument as a C-contiguous NumPy array of T with ndim dimensions
template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> asArray(const py::object& argument, const string& name, int ndim){
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(argument);
    if (!array){
        throw py::type_error(name + " needs a list or a NumPy array of numbers.");
    }
    if (array.ndim() != ndim){
        throw invalid_argument(name + " as an array needs " + to_string(ndim) + " dimensions, not " + to_string(array.ndim()) + ".");
    }
    return array;
}

//the length of dimension k of an array, as a number of states, actions or entries
template <typename A>
int arrayExtent(const A& array, int k, const string& name){
    if (array.shape(k) > INT_MAX){
        throw invalid_argument(name + " is too large: dimension " + to_string(k+1) + " has " + to_string(array.shape(k)) + " entries.");
    }
    return (int)array.shape(k);
}

}

void ModuleInterface::mdp(double discount,
    py::object rewards,
    py::object rewardsElementwise,
    string rewardsFromFile,
    py::object tranMatWithZeros,
    py::object tranMatElementwise,
    py::object tranMatProbs,
    py::object tranMatColumns,
    string tranMatFromFile){
    //select the general MDP problem
    problem.problemType="mdp";
    problem.discount=discount;

    //load the rewards
    if (given(rewards)){
        if (py::isinstance<py::list>(rewards)){
            loadRewardsFromList(rewards.cast<py::list>());
        }else{
            auto array = asArray<double>(rewards,"rewards",2);
            loadRewardsFromArray(array.data(), arrayExtent(array,0,"rewards"), arrayExtent(array,1,"rewards"));
        }
    }else if(given(rewardsElementwise)){
        if (py::isinstance<py::list>(rewardsElementwise)){
            loadRewardsElementwise(rewardsElementwise.cast<py::list>());
        }else{
            auto array = asArray<double>(rewardsElementwise,"rewardsElementwise",2);
            loadRewardsElementwiseArray(array.data(), array.shape(0), array.shape(1));
        }
    }else{
        loadRewardsFromFile(rewardsFromFile,',',true);
    }

    //load the transition probabilities
    if (given(tranMatWithZeros)){
        if (py::isinstance<py::list>(tranMatWithZeros)){
            loadTranMatWithZeros(tranMatWithZeros.cast<py::list>());
        }else{
            auto array = asArray<double>(tranMatWithZeros,"tranMatWithZeros",3);
            loadTranMatWithZerosArray(array.data(), arrayExtent(array,0,"tranMatWithZeros"),
                arrayExtent(array,1,"tranMatWithZeros"), arrayExtent(array,2,"tranMatWithZeros"));
        }
    }else if(given(tranMatElementwise)){
        if (py::isinstance<py::list>(tranMatElementwise)){
            loadTranMatElementwise(tranMatElementwise.cast<py::list>());
        }else{
            auto array = asArray<double>(tranMatElementwise,"tranMatElementwise",2);
            loadTranMatElementwiseArray(array.data(), array.shape(0), array.shape(1));
        }
    }else if(given(tranMatProbs) && given(tranMatColumns)){
        if (py::isinstance<py::list>(tranMatProbs) && py::isinstance<py::list>(tranMatColumns)){
            loadTranMatFromLists(tranMatProbs.cast<py::list>(),tranMatColumns.cast<py::list>());
        }else{
            auto probs = asArray<double>(tranMatProbs,"tranMatProbs",3);
            auto columns = asArray<int64_t>(tranMatColumns,"tranMatColumns",3);
            for (int k=0; k<3; k++){
                if (probs.shape(k) != columns.shape(k)){
                    throw invalid_argument("tranMatProbs and tranMatColumns as arrays need the same shape.");
                }
            }
            loadTranMatFromArrays(probs.data(), columns.data(), arrayExtent(probs,0,"tranMatProbs"),
                arrayExtent(probs,1,"tranMatProbs"), arrayExtent(probs,2,"tranMatProbs"));
        }
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
    settings.postProcessing, settings.makeFinalCheck, settings.parallel);

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

py::array_t<int64_t> ModuleInterface::getPolicy(){
    //a NumPy array of int64 (NumPy's default integer), copied from the policy
    const vector<int>& policy = problem.policy.policy;
    py::array_t<int64_t> result((py::ssize_t)policy.size());
    int64_t* out = result.mutable_data();
    for (size_t sidx=0; sidx<policy.size(); sidx++){
        out[sidx] = policy[sidx];
    }
    return result;
}

py::array_t<double> ModuleInterface::getValueVector(){
    //a NumPy array of float64, copied from the value vector (with no base object given,
    //pybind11 copies the data, so later solves cannot change an array already returned)
    const vector<double>& values = problem.valueVector.valueVector;
    return py::array_t<double>((py::ssize_t)values.size(), values.data());
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

//---------------------------------------
//  LOADING: the nested lists
//---------------------------------------
//The nested lists (rewards, tranMatWithZeros, tranMatProbs and tranMatColumns) are read item
//by item straight into the flat arrays of Rewards and TransitionMatrix, without a nested C++
//copy in between. Lists and tuples are read in place, and any other sequence (e.g. a NumPy
//array as a row) is first turned into a list, as pybind11 accepted them before. Floats and
//ints are read directly; any other item (a NumPy scalar, a bool, ...) is converted by
//pybind11 as before, with the same errors.

namespace {

//the items of a Python sequence
class Items{
public:
    Items() : items(nullptr), n(0) {}
    Items(py::handle sequence, const char* name){
        PyObject* o = sequence.ptr();
        if (PyList_Check(o) || PyTuple_Check(o)){
            fast = py::reinterpret_borrow<py::object>(sequence);
        }else if (PySequence_Check(o) && !PyUnicode_Check(o) && !PyBytes_Check(o)){
            fast = py::reinterpret_steal<py::object>(PySequence_Fast(o, ""));
            if (!fast){
                throw py::error_already_set();
            }
        }else{
            throw py::cast_error(string(name) + " holds a " + Py_TYPE(o)->tp_name + " where a list is needed.");
        }
        n = PySequence_Fast_GET_SIZE(fast.ptr());
        items = PySequence_Fast_ITEMS(fast.ptr());
    }
    Py_ssize_t size() const { return n; }
    py::handle operator[](Py_ssize_t i) const { return items[i]; }
private:
    py::object fast;
    PyObject** items;
    Py_ssize_t n;
};

inline double toDouble(py::handle item){
    if (PyFloat_Check(item.ptr())){
        return PyFloat_AS_DOUBLE(item.ptr());
    }
    return item.cast<double>();
}

inline int toInt(py::handle item){
    if (PyLong_CheckExact(item.ptr())){
        int overflow;
        long value = PyLong_AsLongAndOverflow(item.ptr(), &overflow);
        if (!overflow && value >= INT_MIN && value <= INT_MAX){
            return (int)value;
        }
    }
    return item.cast<int>(); //raises pybind11's error, as before
}

}

void ModuleInterface::loadRewardsFromList(py::list rewardsList){
    Rewards& rewards = problem.rewards;
    rewards.clear();
    Items states(rewardsList,"rewards");
    for (Py_ssize_t sidx=0; sidx<states.size(); sidx++){
        Items actions(states[sidx],"rewards");
        for (Py_ssize_t aidx=0; aidx<actions.size(); aidx++){
            rewards.addReward(toDouble(actions[aidx]));
        }
        rewards.endState();
    }
}

void ModuleInterface::loadTranMatFromLists(py::list tranMatProbs, py::list tranMatColumns){
    //A pair keeps as many entries as both lists have for it, and the actions and states that
    //tranMatColumns lacks have none; checkModel() reports where the two lists differ in shape,
    //from the record made here.
    TransitionMatrix& tranMat = problem.tranMat;
    tranMat.clear();
    Items probStates(tranMatProbs,"tranMatProbs");
    Items columnStates(tranMatColumns,"tranMatColumns");
    Py_ssize_t nStates = probStates.size();
    Py_ssize_t nColumnStates = columnStates.size();
    tranMat.columnRows = (int)nColumnStates;

    //the sizes first, so that the arrays are allocated once
    int64_t nPairs = 0, nEntries = 0;
    for (Py_ssize_t sidx=0; sidx<nStates; sidx++){
        Items probActions(probStates[sidx],"tranMatProbs");
        nPairs += probActions.size();
        if (sidx < nColumnStates){
            Items columnActions(columnStates[sidx],"tranMatColumns");
            for (Py_ssize_t aidx=0; aidx<min(probActions.size(),columnActions.size()); aidx++){
                nEntries += min(Items(probActions[aidx],"tranMatProbs").size(), Items(columnActions[aidx],"tranMatColumns").size());
            }
        }
    }
    tranMat.reserve(nPairs,nEntries);

    for (Py_ssize_t sidx=0; sidx<nStates; sidx++){
        Items probActions(probStates[sidx],"tranMatProbs");
        Items columnActions;
        if (sidx < nColumnStates){
            columnActions = Items(columnStates[sidx],"tranMatColumns");
            if (columnActions.size() != probActions.size()){
                tranMat.shapeMismatches.push_back({(int)sidx, -1, (int)probActions.size(), (int)columnActions.size()});
            }
        }
        for (Py_ssize_t aidx=0; aidx<probActions.size(); aidx++){
            if (aidx < columnActions.size()){
                Items probs(probActions[aidx],"tranMatProbs");
                Items columns(columnActions[aidx],"tranMatColumns");
                if (probs.size() != columns.size()){
                    tranMat.shapeMismatches.push_back({(int)sidx, (int)aidx, (int)probs.size(), (int)columns.size()});
                }
                for (Py_ssize_t cidx=0; cidx<min(probs.size(),columns.size()); cidx++){
                    tranMat.addEntry(toInt(columns[cidx]), toDouble(probs[cidx]));
                }
            }
            tranMat.endPair();
        }
        tranMat.endState();
    }
}

void ModuleInterface::loadTranMatWithZeros(py::list tranMatWithZeros){
    //the probabilities above zero, with their column indices as next states
    TransitionMatrix& tranMat = problem.tranMat;
    tranMat.clear();
    Items states(tranMatWithZeros,"tranMatWithZeros");
    for (Py_ssize_t sidx=0; sidx<states.size(); sidx++){
        Items actions(states[sidx],"tranMatWithZeros");
        for (Py_ssize_t aidx=0; aidx<actions.size(); aidx++){
            Items row(actions[aidx],"tranMatWithZeros");
            for (Py_ssize_t jidx=0; jidx<row.size(); jidx++){
                double prob = toDouble(row[jidx]);
                if (prob > 0.0){
                    tranMat.addEntry((int)jidx, prob);
                }
            }
            tranMat.endPair();
        }
        tranMat.endState();
    }
    tranMat.shrink();
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
    for (int sidx=0; sidx<numberOfStates; sidx++){
        nAct[sidx]++; //from the highest action index to the number of actions
    }
    problem.rewards.setActions(nAct);
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
    for (int sidx=0; sidx<numberOfStates; sidx++){
        nAct[sidx]++; //from the highest action index to the number of actions
    }
    problem.rewards.setActions(nAct);

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
    int k0,k1,k2;
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
    //determine number of actions and columns (next states) in each state
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = tranMatElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = listIndex(innerRew,1,i,name,"action");
        if (k1>nAct[k0]){
            nAct[k0]=k1;
        }
    }
    for (int sidx=0; sidx<numberOfStates; sidx++){
        nAct[sidx]++; //from the highest action index to the number of actions
    }
    problem.tranMat.setActions(nAct);
    //count the entries of each state-action pair
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = tranMatElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = innerRew[1].cast<int>();
        problem.tranMat.countEntry(k0,k1);
    }
    //allocate memory
    problem.tranMat.allocateEntries();
    //assign values, each to the next free entry of its pair (in the order of the list)
    for (int i=0; i<tranMatElementwise.size(); i++){
        innerRew = tranMatElementwise[i].cast<py::list>();
        k0 = innerRew[0].cast<int>(); k1 = innerRew[1].cast<int>();
        k2 = listIndex(innerRew,2,i,name,"next-state"); prob = innerRew[3].cast<double>();
        problem.tranMat.fillEntry(k0,k1,k2,prob);
    }
    problem.tranMat.finishEntries();
}


void ModuleInterface::loadTranMatFromFile(string tranMatFromFile, char sep, bool header){
    
    const string columns = "state, action, next state, probability";
    string line;
    vector<string> fields;
    vector<int> nAct;
    int i,k0,k1,k2;
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
        nAct[sidx]++; //from the highest action index to the number of actions
    }
    problem.tranMat.setActions(nAct);

    //reset
    file.clear();
    file.seekg(0,ios::beg);

    //count the entries of each state-action pair
    i=0;
    while (getline(file,line)){
        if ((!header||i>0) && fileRow(line,sep,4,tranMatFromFile,i+1,columns,fields)){
            k0 = stoi(fields[0]); k1 = stoi(fields[1]);
            problem.tranMat.countEntry(k0,k1);
        }
        i++;
    }

    //reset
    file.clear();
    file.seekg(0,ios::beg);

    //allocate memory
    problem.tranMat.allocateEntries();

    //assign values, each to the next free entry of its pair (in the order of the file)
    i=0;
    while (getline(file,line)){
        if ((!header||i>0) && fileRow(line,sep,4,tranMatFromFile,i+1,columns,fields)){
            k0 = stoi(fields[0]); k1 = stoi(fields[1]);
            k2 = fileIndex(fields[2],tranMatFromFile,i+1,"next-state");
            prob = fileNumber(fields[3],tranMatFromFile,i+1,"probability");
            problem.tranMat.fillEntry(k0,k1,k2,prob);
        }
        i++;
    }
    file.close();
    problem.tranMat.finishEntries();
}


//---------------------------------------
//  LOADING: NumPy arrays
//---------------------------------------
//The same formats as the lists, read from the data of NumPy arrays (C-contiguous, row by
//row). These loaders use no Python, so that they stay the same whatever binds them. They
//keep the order of the entries and the crash guards of the list loaders: an elementwise
//row's state, action and next state must be whole numbers and not negative.

namespace {

//a number as text: -1 as "-1", 1.5 as "1.5"
string numberText(double x){
    ostringstream text;
    text << x;
    return text.str();
}

//an index in row i of an elementwise array: a whole number, not negative
int arrayIndex(double x, int64_t i, const string& name, const string& what){
    if (!(x == floor(x))){ //also NaN and infinity
        throw invalid_argument(name + "[" + to_string(i) + "] has the " + what + " index " + numberText(x) + ", which is not an integer.");
    }
    if (x < 0){
        throw invalid_argument(name + "[" + to_string(i) + "] has the negative " + what + " index " + numberText(x) + ".");
    }
    if (x > INT_MAX){
        throw invalid_argument(name + "[" + to_string(i) + "] has the " + what + " index " + numberText(x) + ", which is too large.");
    }
    return (int)x;
}

//an elementwise array needs at least n columns
void checkRowLength(int64_t rowLength, int64_t n, const string& name, const string& columns){
    if (rowLength < n){
        throw invalid_argument(name + " has " + to_string(rowLength) + " columns, where " + to_string(n) + " are needed (" + columns + ").");
    }
}

}

void ModuleInterface::loadRewardsFromArray(const double* rewardsArray, int nStates, int nActions){
    //rewards[s][a] = rewardsArray[s*nActions + a]
    Rewards& rewards = problem.rewards;
    rewards.clear();
    rewards.reserve((int64_t)nStates * nActions);
    for (int sidx=0; sidx<nStates; sidx++){
        for (int aidx=0; aidx<nActions; aidx++){
            rewards.addReward(rewardsArray[(int64_t)sidx*nActions + aidx]);
        }
        rewards.endState();
    }
}

void ModuleInterface::loadRewardsElementwiseArray(const double* rows, int64_t nRows, int64_t rowLength){
    //row i: state, action, reward (as for the list, the last reward of a pair counts)
    const string name = "rewardsElementwise";
    checkRowLength(rowLength,3,name,"state, action, reward");
    int numberOfStates=0;
    for (int64_t i=0; i<nRows; i++){
        numberOfStates = max(numberOfStates, arrayIndex(rows[i*rowLength],i,name,"state") + 1);
    }
    vector<int> nAct(numberOfStates,1);
    for (int64_t i=0; i<nRows; i++){
        int k0 = (int)rows[i*rowLength];
        nAct[k0] = max(nAct[k0], arrayIndex(rows[i*rowLength+1],i,name,"action") + 1);
    }
    problem.rewards.setActions(nAct);
    for (int64_t i=0; i<nRows; i++){
        problem.rewards.assignReward(rows[i*rowLength+2], (int)rows[i*rowLength], (int)rows[i*rowLength+1]);
    }
}

void ModuleInterface::loadTranMatWithZerosArray(const double* probs, int nStates, int nActions, int nColumns){
    //the probabilities above zero of probs[s][a][j] = probs[(s*nActions + a)*nColumns + j],
    //with their column indices j as next states
    TransitionMatrix& tranMat = problem.tranMat;
    tranMat.clear();
    int64_t nEntries = 0, nElements = (int64_t)nStates * nActions * nColumns;
    for (int64_t k=0; k<nElements; k++){
        if (probs[k] > 0.0){
            nEntries++;
        }
    }
    tranMat.reserve((int64_t)nStates * nActions, nEntries);
    for (int sidx=0; sidx<nStates; sidx++){
        for (int aidx=0; aidx<nActions; aidx++){
            const double* row = probs + ((int64_t)sidx*nActions + aidx) * nColumns;
            for (int jidx=0; jidx<nColumns; jidx++){
                if (row[jidx] > 0.0){
                    tranMat.addEntry(jidx, row[jidx]);
                }
            }
            tranMat.endPair();
        }
        tranMat.endState();
    }
}

void ModuleInterface::loadTranMatElementwiseArray(const double* rows, int64_t nRows, int64_t rowLength){
    //row i: state, action, next state, probability; each entry goes to the next free place
    //of its pair, in the order of the rows
    const string name = "tranMatElementwise";
    checkRowLength(rowLength,4,name,"state, action, next state, probability");
    int numberOfStates=0;
    for (int64_t i=0; i<nRows; i++){
        numberOfStates = max(numberOfStates, arrayIndex(rows[i*rowLength],i,name,"state") + 1);
    }
    vector<int> nAct(numberOfStates,1);
    for (int64_t i=0; i<nRows; i++){
        int k0 = (int)rows[i*rowLength];
        nAct[k0] = max(nAct[k0], arrayIndex(rows[i*rowLength+1],i,name,"action") + 1);
    }
    TransitionMatrix& tranMat = problem.tranMat;
    tranMat.setActions(nAct);
    for (int64_t i=0; i<nRows; i++){
        tranMat.countEntry((int)rows[i*rowLength], (int)rows[i*rowLength+1]);
    }
    tranMat.allocateEntries();
    for (int64_t i=0; i<nRows; i++){
        const double* row = rows + i*rowLength;
        tranMat.fillEntry((int)row[0], (int)row[1], arrayIndex(row[2],i,name,"next-state"), row[3]);
    }
    tranMat.finishEntries();
}

void ModuleInterface::loadTranMatFromArrays(const double* probs, const int64_t* columns, int nStates, int nActions, int nEntries){
    //nEntries probabilities and next states per pair, all kept (a padding probability of 0
    //adds nothing to a value)
    TransitionMatrix& tranMat = problem.tranMat;
    tranMat.clear();
    tranMat.reserve((int64_t)nStates * nActions, (int64_t)nStates * nActions * nEntries);
    int64_t k = 0;
    for (int sidx=0; sidx<nStates; sidx++){
        for (int aidx=0; aidx<nActions; aidx++){
            for (int cidx=0; cidx<nEntries; cidx++, k++){
                if (columns[k] < INT_MIN || columns[k] > INT_MAX){
                    throw invalid_argument("tranMatColumns holds the column index " + to_string(columns[k]) + ", which is too large.");
                }
                tranMat.addEntry((int)columns[k], probs[k]);
            }
            tranMat.endPair();
        }
        tranMat.endState();
    }
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

    //where tranMatProbs and tranMatColumns differ in shape, as recorded while loading them
    //(in the order of the loop below); a pair kept as many entries as both lists have
    const vector<TransitionMatrix::ShapeMismatch>& mismatches = tranMat.shapeMismatches;
    size_t nextMismatch = 0;

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
        int nColumnActions = nActions;
        if (nextMismatch < mismatches.size() && mismatches[nextMismatch].sidx == sidx && mismatches[nextMismatch].aidx == -1){
            nColumnActions = mismatches[nextMismatch].nColumns;
            nextMismatch++;
            shapeMismatch.add("state " + to_string(sidx) + " (" + counted(nActions,"action","actions") + " in tranMatProbs, " + to_string(nColumnActions) + " in tranMatColumns)");
        }

        for (int aidx=0; aidx<min(nActions,nColumnActions); aidx++){
            if (nextMismatch < mismatches.size() && mismatches[nextMismatch].sidx == sidx && mismatches[nextMismatch].aidx == aidx){
                const TransitionMatrix::ShapeMismatch& m = mismatches[nextMismatch];
                nextMismatch++;
                shapeMismatch.add(stateAction(sidx,aidx) + " (" + counted(m.nProbs,"entry","entries") + " in tranMatProbs, " + to_string(m.nColumns) + " in tranMatColumns)");
            }
            int nJumps = tranMat.numberOfColumns(sidx,aidx);
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
