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

#include "Rewards.h"

Rewards::Rewards() {
}

Rewards::Rewards(const Rewards& orig) {
}

Rewards::~Rewards() {
}

double Rewards::getReward(int& sidx, int& aidx){
    return rewards[sidx][aidx];
}

void Rewards::assignReward(double reward, int& sidx, int& aidx){
    //assign single probability
    rewards[sidx][aidx]=reward;
} 

void Rewards::assignRewardsFromList(py::list pyRewards){
    //cast rewards directly from Python list
    rewards=pyRewards.cast<vector<vector<double>>>();
} 
    
void Rewards::setNumberOfRows(int numberOfStates){
    rewards.resize(numberOfStates);
}

void Rewards::setNumberOfActions(int nActions, int& sidx){
    rewards[sidx].resize(nActions,-1);
}
    
int Rewards::numberOfActions(int& sidx){
    return rewards[sidx].size();
}

int Rewards::numberOfRows(){
    return rewards.size();
}

