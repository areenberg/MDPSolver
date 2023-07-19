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
        .def("tbm", &ModuleInterface::tbm,"Selects the TBM model.",py::arg("discount")=0.99,py::arg("components")=2,py::arg("stages")=10)
        .def("cbm", &ModuleInterface::cbm,"Selects the CBM model.",py::arg("discount")=0.99,py::arg("components")=2,py::arg("stages")=10,py::arg("pCompMat"))
        .def("solve", &ModuleInterface::solve,"Solves the policy")
        .def("printPolicy", &ModuleInterface::printPolicy,"Prints the entire policy.");

}