#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>

#include "ModuleInterface.h"

using namespace std;
namespace py = pybind11;


PYBIND11_MODULE(solvermodule, m) {

 py::class_<ModuleInterface>(m, "Model")
        .def(py::init<>())
        .def("mdp", &ModuleInterface::mdp,"Selects the general MDP model.", //GENERAL MDP
        py::arg("discount")=0.99,
        py::arg("rewards")=py::list(),
        py::arg("rewardsElementwise")=py::list(),
        py::arg("rewardsFromFile")="rewards.csv",
        py::arg("tranMatWithZeros")=py::list(),
        py::arg("tranMatElementwise")=py::list(),
        py::arg("tranMatProbs")=py::list(),
        py::arg("tranMatColumns")=py::list(),
        py::arg("tranMatFromFile")="transitions.csv")
        .def("tbm", &ModuleInterface::tbm,"Selects the TBM model.", //TBM MODEL
        py::arg("discount")=0.99,
        py::arg("components")=2,
        py::arg("stages")=10,
        py::arg("replacementCost")=-10,
        py::arg("setupCost")=-10,
        py::arg("unexpectedFailureCost")=-20,
        py::arg("expiredNotFixedCost")=-1e6,
        py::arg("failureProb")=0.1,
        py::arg("failureProbMin")=0.01,
        py::arg("failureProbHat")=0.1)
        .def("cbm", &ModuleInterface::cbm,"Selects the CBM model.", //CBM MODEL
        py::arg("discount")=0.99,
        py::arg("components")=2,
        py::arg("stages")=10,
        py::arg("pCompMat")=py::list(),
        py::arg("preventiveCost")=-5,
        py::arg("correctiveCost")=-11,
        py::arg("setupCost")=-4,
        py::arg("failurePenalty")=-300,
        py::arg("kOfN")=-1)
        .def("solve", &ModuleInterface::solve,"Solves the policy", //SOLVE
        py::arg("algorithm")="mpi",
        py::arg("tolerance")=1e-3,
        py::arg("update")="standard",
        py::arg("parIterLim")=100,
        py::arg("SORrelaxation")=1.0,
        py::arg("initPolicy")=py::list(),
        py::arg("initValueVector")=py::list(),
        py::arg("verbose")=false,
        py::arg("postProcessing")=true,
        py::arg("makeFinalCheck")=true)
        .def("getRuntime",&ModuleInterface::getRuntime,"Returns the runtime in milliseconds.") //OUTPUT
        .def("printPolicy", &ModuleInterface::printPolicy,"Prints the entire policy.")
        .def("printValueVector", &ModuleInterface::printValueVector,"Prints the entire value vector.")
        .def("getAction", &ModuleInterface::getAction,"Returns an action index from the optimized policy.",py::arg("stateIndex")=0)
        .def("getValue", &ModuleInterface::getValue,"Returns a value from the optimized policy.",py::arg("stateIndex")=0)
        .def("getPolicy", &ModuleInterface::getPolicy,"Returns the optimized policy.")
        .def("getValueVector", &ModuleInterface::getValueVector,"Returns the optimized value vector.")
        .def("saveToFile", &ModuleInterface::saveToFile,"Saves the optimized policy or value vector to a file.",py::arg("fileName")="result.csv",py::arg("type")="policy");

}