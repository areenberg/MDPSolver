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

#include "GeneralMDPmodel.h"

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
    numberOfStates=tranMat->numberOfRows();
    cidx=0;
}
        
double GeneralMDPmodel::reward(int &sidx, int &aidx){
    return rewards->getReward(sidx,aidx);
}

double GeneralMDPmodel::transProb(int &sidx, int &aidx, int &jidx){
    //calculates the probability of jumping to state jidx from
    //the current state sidx when taking action aidx.
    //returns *and* stores the calculated probability in the variable psj.

    return tranMat->getProb(sidx,aidx,jidx);
}

int GeneralMDPmodel::getNumberOfJumps(int &sidx, int &aidx){
    return tranMat->numberOfColumns(sidx,aidx);
}

void GeneralMDPmodel::updateNextState(int &sidx, int &aidx, int &jidx){
    //updates the next possible state, nextState, and the associated
    //transition probability, psj.    
    while (tranMat->getColumn(sidx,aidx,cidx)!=jidx){ //this will in many cases immediately evaluate to False
        cidx++;
        if (cidx==tranMat->numberOfColumns(sidx,aidx)){ //to make sure cidx is always feasible
            cidx=0;
        }    
    }
    cidx++; //move one additional step
    if (cidx==tranMat->numberOfColumns(sidx,aidx)){
        cidx=0;
    }
    nextState = tranMat->getColumn(sidx,aidx,cidx);
    psj = tranMat->getProb(sidx,aidx,cidx);
}    

int GeneralMDPmodel::getColumnIdx(int &sidx, int &aidx, int &cidx){
    return tranMat->getColumn(sidx,aidx,cidx);
}

int GeneralMDPmodel::postDecisionIdx(int &sidx, int &aidx){
    //derives the first new/next state that is possible
    //to reach from the current state sidx.
    //both returns the value of the first new state, and
    //stores it in the variable nextState.
    cidx=0;
    nextState = tranMat->getColumn(sidx,aidx,cidx);
    return nextState;
}

double GeneralMDPmodel::getDiscount(){
    return discount;
}

int GeneralMDPmodel::getNumberOfStates(){
    return numberOfStates;
}

void GeneralMDPmodel::updateNumberOfActions(int &sidx){
    numberOfActions=tranMat->numberOfActions(sidx);
}

int GeneralMDPmodel::getNumberOfActions(){
    return numberOfActions;
}

int GeneralMDPmodel::getNumberOfActions(int &sidx){
    return tranMat->numberOfActions(sidx);
}

int * GeneralMDPmodel::getNextState(){
    return &nextState;
}

double GeneralMDPmodel::getPsj(){
    return psj;
}
