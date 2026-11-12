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

#include "GeneralMDPmodel.h"
#include <limits>

GeneralMDPmodel::GeneralMDPmodel(Rewards * rw, TransitionMatrix * tm, double discount):
rewards(rw),
tranMat(tm),
discount(discount)
{
    initialize();
}

GeneralMDPmodel::GeneralMDPmodel(const GeneralMDPmodel& orig) {
}

GeneralMDPmodel::~GeneralMDPmodel() {
}


void GeneralMDPmodel::initialize(){
    tranMat->fillTables(csr);
    numberOfStates=csr.numberOfStates;
    cidx=0;

    //The tables need one reward per pair of the transition probabilities. In a well-formed
    //model the rewards have the same numbers of actions, and so the same numbering, and
    //their array is used as it is. Otherwise (which checkModel() reports) a copy numbered
    //like the pairs is made, so that the solver reads no memory outside the rewards.
    bool aligned = rewards->numberOfRows() >= numberOfStates;
    for (int sidx=0; aligned && sidx<numberOfStates; sidx++){
        aligned = rewards->numberOfActions(sidx) == tranMat->numberOfActions(sidx);
    }
    if (aligned){
        csr.reward = rewards->data();
    }else{
        alignedRewards.assign(csr.stateStart[numberOfStates], numeric_limits<double>::quiet_NaN());
        for (int sidx=0; sidx<numberOfStates && sidx<rewards->numberOfRows(); sidx++){
            for (int aidx=0; aidx<tranMat->numberOfActions(sidx) && aidx<rewards->numberOfActions(sidx); aidx++){
                alignedRewards[pair(sidx,aidx)] = rewards->getReward(sidx,aidx);
            }
        }
        csr.reward = alignedRewards.data();
    }
}

double GeneralMDPmodel::reward(int &sidx, int &aidx){
    //calculates and returns the reward of taking action
    //aidx when in current state sidx

    return csr.reward[pair(sidx,aidx)];
}

double GeneralMDPmodel::transProb(int &sidx, int &aidx, int &jidx){
    //returns the jidx-th non-zero transition probability of taking
    //action aidx in state sidx (jidx counts the non-zero probabilities)

    return csr.prob[entry(sidx,aidx,jidx)];
}

int GeneralMDPmodel::getNumberOfJumps(int &sidx, int &aidx){
    int64_t i = pair(sidx,aidx);
    return (int)(csr.pairStart[i+1] - csr.pairStart[i]);
}

void GeneralMDPmodel::updateNextState(int &sidx, int &aidx, int &jidx){
    //updates the next possible state, nextState, and the associated
    //transition probability, psj.
    int nJumps = getNumberOfJumps(sidx,aidx);
    while (csr.column[entry(sidx,aidx,cidx)]!=jidx){ //this will in many cases immediately evaluate to False
        cidx++;
        if (cidx==nJumps){ //to make sure cidx is always feasible
            cidx=0;
        }
    }
    cidx++; //move one additional step
    if (cidx==nJumps){
        cidx=0;
    }
    nextState = csr.column[entry(sidx,aidx,cidx)];
    psj = csr.prob[entry(sidx,aidx,cidx)];
}

int GeneralMDPmodel::getColumnIdx(int &sidx, int &aidx, int &cidx){
    return csr.column[entry(sidx,aidx,cidx)];
}

int GeneralMDPmodel::postDecisionIdx(int &sidx, int &aidx){
    //derives the first new/next state that is possible
    //to reach from the current state sidx.
    //both returns the value of the first new state, and
    //stores it in the variable nextState.
    cidx=0;
    nextState = csr.column[entry(sidx,aidx,cidx)];
    return nextState;
}

double GeneralMDPmodel::getDiscount(){
    return discount;
}

int GeneralMDPmodel::getNumberOfStates(){
    return numberOfStates;
}

void GeneralMDPmodel::updateNumberOfActions(int &sidx){
    numberOfActions=(int)(csr.stateStart[sidx+1] - csr.stateStart[sidx]);
}

int GeneralMDPmodel::getNumberOfActions(){
    return numberOfActions;
}

int GeneralMDPmodel::getNumberOfActions(int &sidx){
    return (int)(csr.stateStart[sidx+1] - csr.stateStart[sidx]);
}

int * GeneralMDPmodel::getNextState(){
    return &nextState;
}

double GeneralMDPmodel::getPsj(){
    return psj;
}
