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

#include "TransitionMatrix.h"

TransitionMatrix::TransitionMatrix() {
}

TransitionMatrix::TransitionMatrix(const TransitionMatrix& orig) {
}

TransitionMatrix::~TransitionMatrix() {
}

double TransitionMatrix::getProb(int& sidx, int& aidx, int& cidx){
    return probs[sidx][aidx][cidx];
}
 
int TransitionMatrix::getColumn(int& sidx, int& aidx, int& cidx){
    return cols[sidx][aidx][cidx];
}

void TransitionMatrix::assignProb(double prob, int& sidx, int& aidx, int& cidx){
    probs[sidx][aidx][cidx]=prob; 
}

void TransitionMatrix::assignColumn(int column, int& sidx, int& aidx, int& cidx){
    cols[sidx][aidx][cidx]=column;
}

void TransitionMatrix::assignProbsFromList(py::list pyProbs){ 
    //cast probabilities directly from Python list
    probs=pyProbs.cast<vector<vector<vector<double>>>>();
}
    
void TransitionMatrix::assignColumnsFromList(py::list pyCols){ 
    //cast column indices directly from Python list
    cols=pyCols.cast<vector<vector<vector<int>>>>();
}    
    
void TransitionMatrix::setNumberOfRows(int numberOfStates){
    probs.resize(numberOfStates);
    cols.resize(numberOfStates);
}

void TransitionMatrix::setNumberOfActions(int nActions, int& sidx){
    probs[sidx].resize(nActions);
    cols[sidx].resize(nActions);
}

void TransitionMatrix::setNumberOfColumns(int nJumps, int& sidx, int& aidx){
    probs[sidx][aidx].resize(nJumps,-1);
    cols[sidx][aidx].resize(nJumps,-1);
}

int TransitionMatrix::numberOfColumns(int& sidx, int& aidx){
    return cols[sidx][aidx].size();
}

int TransitionMatrix::numberOfActions(int& sidx){
    return probs[sidx].size();
}

int TransitionMatrix::numberOfRows(){
    return probs.size();
}