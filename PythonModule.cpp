#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

#include "ModelType.h" //Generic model class
#include "TBMmodel.h" //Time-based maintenance model
#include "CBMmodel.h" //Condition-based maintenance model
#include "ModifiedPolicyIteration.h" //The solver

using namespace std;
namespace py = pybind11;

//TO DO:
//- create a class for the policy (including the value vector)
//- create a class for the transition probabilities
//- create a class for the reward function
//- create the generic model class

//---------------------------------------
//  OBJECT POINTERS
//---------------------------------------

//Might not be necesarry
//- insert pointers to:
//  * transitions
//  * rewards
//  * policy and value vector

//model and solver object is deleted after computations finished

//---------------------------------------
//  DATA STRUCTURES
//---------------------------------------

//problem settings
struct Problem{
    string minmax;
    double discount;

    //only for the TBM/CBM problem
    int nComp;
    int compStates;
    vector<vector<double>> pCompMat; //only for CBM
} problem;


//solver settings
struct Settings{
    string algorithm="mpi";
    double eps=1e-3;
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

void selectGeneric(double discount, double minmax, py::list transProbs,
 py::list transIdx, py::list rewards, 
 string pathToTransProbs, string pathToTransIdx,string pathToRewards){
    
    //store parameters and settings (model object is created in the solve method)

}

void selectTBM(double discount,int nComp,int lifetime){

    //store parameters and settings (model object is created in the solve method)

}

void selectCBM(double discount,int nComp,int conditions,
string pCompMat){

    //store parameters and settings (model object is created in the solve method)

}

//check and solve

void checkParameters(){
    //checks if parameters are feasible
}

void solve(){

    //create model from stored parameters
    
    //solve the model

    //results will be saved in the policy object
    
}




//---------------------------------------
//  INTERFACE TO PYTHON
//---------------------------------------


PYBIND11_MODULE(relsys, m) {
    
    //select the optimization problem
    m.def("mdp",&selectGeneric,"Selects the generic MDP.",py::arg("discount")=0.99,py::arg("minmax")="min",
    py::arg("transProbs"),py::arg("transIdx"),py::arg("rewards"),py::arg("pathToTransProbs")=" ",py::arg("pathToTransIdx")=" ",py::arg("pathToRewards")=" ");
    m.def("tbm",&selectTBM,"Selects the TBM model.",py::arg("discount")=0.99,py::arg("nComp")=2,py::arg("lifetime")=10);
    m.def("cbm",&selectCBM,"Selects the CBM model.",py::arg("discount")=0.99,py::arg("nComp")=2,py::arg("conditions")=10,py::arg("pCompMat")="pCompMat.csv");


    //model settings
    m.def("solve",&solve,"Solves the policy"); 

        
}
