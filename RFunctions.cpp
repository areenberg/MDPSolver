#include <Rcpp.h>
#include <limits.h>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>

#include "CBMmodel.h"
#include "TBMmodel.h"
#include "modifiedPolicyIteration.h"


using namespace Rcpp;



// [[Rcpp::export]]
List tbm(int components=2,int timePeriods=5,
        double discount=0.95, double eps=1e-6, 
        string algorithm = "MPI", string update = "Standard",
        int parIterLim = 100, bool useSpan=true, double SORrelaxation = 1.0,
        int limitOutput=R_PosInf){
    //solves the time-based component replacement problem

    
    //create the model object
    Model mdl(components, timePeriods, discount);
    
    //create the solver object
    modifiedPolicyIteration mpi(mdl,eps,algorithm,update,
        parIterLim,SORrelaxation);
    
    //solve the problem
    mpi.solve(mdl);
    
    
    //insert solution into output list
    int lm;
    if (!is_infinite(limitOutput)){
        lm=limitOutput;
    }else{
        lm=mpi.v.size();
    }

    NumericVector policyVector(lm);
    NumericVector valueVector(lm);

    for (int sidx=0; sidx<lm; sidx++){
        policyVector[sidx]=mdl.policy[sidx];
        valueVector[sidx]=mpi.v[sidx];
    }
    
    List results = List::create(Named("Policy") = policyVector , _["Values"] = valueVector, 
            _["Iterations"] = mpi.iter, _["Duration"] = mpi.duration);
    
    return(results);   
}



// [[Rcpp::export]]
void cbm(NumericMatrix pMat, int components=2,int timePeriods=5,
        double discount=0.95, double eps=1e-6, 
        string algorithm = "MPI", string update = "Standard",
        int parIterLim = 100, bool useSpan=true, double SORrelaxation = 1.0,
        int limitOutput=R_PosInf){
    //solves the condition-based component replacement problem

    //pMat is the transition probability matrix    

}




// [[Rcpp::export]]
void mdp(){
    //solves a Markov Decision Process with custom reward function, action
    //space, and transition probabilities
    
    
    
}







