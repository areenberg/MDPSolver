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

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

#ifndef REWARDS_H
#define REWARDS_H

class Rewards {
public:
    
    Rewards();
    Rewards(const Rewards& orig);
    virtual ~Rewards();
    
    //METHODS
    
    //read and write values
    double * getReward(int& sidx, int& aidx);
    void assignReward(double reward, int& sidx, int& aidx); //assign single probability
    void assignRewardsFromList(py::list pyRewards); //cast probabilities directly from Python list
    
    //set size of array
    void setNumberOfRows(int numberOfStates);
    void setNumberOfActions(int nActions, int& sidx);
    
    int numberOfActions(int& sidx);
    int numberOfRows();
    
private:

    //VARIABLES
    vector<vector<double>> rewards; //array of rewards (index1: state, index2: action)
    
};

#endif /* REWARDS_H */

