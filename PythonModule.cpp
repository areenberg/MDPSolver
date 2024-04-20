#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>

#include "ModuleInterface.h"

using namespace std;
namespace py = pybind11;


PYBIND11_MODULE(mdpsolver, m) {

 py::class_<ModuleInterface>(m, "Model")
        .def(py::init<>())
        .def("mdp", &ModuleInterface::mdp,"Selects the general MDP model.",
        py::arg("discount")=0.99,
        py::arg("rewards")=py::list(),
        py::arg("rewardsElementwise")=py::list(),
        py::arg("rewardsFromFile")="rewards.csv",
        py::arg("tranMatWithZeros")=py::list(),
        py::arg("tranMatElementwise")=py::list(),
        py::arg("tranMatProbs")=py::list(),
        py::arg("tranMatColumns")=py::list(),
        py::arg("tranMatFromFile")="transitions.csv")
        .def("tbm", &ModuleInterface::tbm,"Selects the TBM model.",py::arg("discount")=0.99,py::arg("components")=2,py::arg("stages")=10)
        .def("cbm", &ModuleInterface::cbm,"Selects the CBM model.",py::arg("discount")=0.99,py::arg("components")=2,py::arg("stages")=10,py::arg("pCompMat"))
        .def("solve", &ModuleInterface::solve,"Solves the policy",
        py::arg("algorithm")="mpi",
        py::arg("tolerance")=1e-3,
        py::arg("update")="standard",
        py::arg("parIterLim")=100,
        py::arg("SORrelaxation")=1.0,
        py::arg("verbose")=false)
        .def("printPolicy", &ModuleInterface::printPolicy,"Prints the entire policy.")
        .def("printValueVector", &ModuleInterface::printValueVector,"Prints the entire value vector.")
        .def("getAction", &ModuleInterface::getAction,"Returns an action index from the optimized policy.",py::arg("stateIndex")=0)
        .def("getValue", &ModuleInterface::getValue,"Returns a value from the optimized policy.",py::arg("stateIndex")=0)
        .def("getPolicy", &ModuleInterface::getPolicy,"Returns the optimized policy.")
        .def("getValueVector", &ModuleInterface::getValueVector,"Returns the optimized value vector.")
        .def("saveToFile", &ModuleInterface::saveToFile,"Saves the optimized policy or value vector to a file.",py::arg("fileName")="result.csv",py::arg("type")="policy");

}