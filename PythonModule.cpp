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

//TO DO:
//- create a class for the transition probabilities
//- create a class for the reward function
//- create the generic model class

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

//problem selection

//void mdp(double discount, double minmax, py::list transProbs,
// py::list transIdx, py::list rewards, 
// string pathToTransProbs, string pathToTransIdx,string pathToRewards){
    
    //store parameters and settings (model object is created in the solve method)

//}

void tbm(double discount,int components,int stages){
    //selects the TBM problem
    problem.problemType="tbm";
    problem.discount=discount;
    problem.components=components;
    problem.stages=stages;
    cout << "Selected time-based maintenance problem with " << problem.components <<
     " components and " << problem.stages << " stages." << endl;
}

void cbm(double discount,int components,int stages,
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

//check and solve

void checkParameters(){
    //checks if parameters are feasible
}

void solve(){

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

void printPolicy(){
    for (int sidx=0; sidx<problem.policy.policy.size(); sidx++){
        cout << (sidx+1) << ": " << problem.policy.policy[sidx] << endl; 
    }
}



//---------------------------------------
//  INTERFACE TO PYTHON
//---------------------------------------


PYBIND11_MODULE(mdpsolver, m) {
    
    //select the optimization problem
    //m.def("mdp",&mdp,"Selects the generic MDP.",py::arg("discount")=0.99,py::arg("minmax")="min",
    //py::arg("transProbs"),py::arg("transIdx"),py::arg("rewards"),py::arg("pathToTransProbs")=" ",py::arg("pathToTransIdx")=" ",py::arg("pathToRewards")=" ");
    m.def("tbm",&tbm,"Selects the TBM model.",py::arg("discount")=0.99,py::arg("components")=2,py::arg("stages")=10);
    m.def("cbm",&cbm,"Selects the CBM model.",py::arg("discount")=0.99,py::arg("components")=2,py::arg("stages")=10,py::arg("pCompMat"));


    //model settings
    m.def("solve",&solve,"Solves the policy"); 
        

   //get results
   m.def("printPolicy",&printPolicy,"Prints the entire policy."); 
    

}
