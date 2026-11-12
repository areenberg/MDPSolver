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

#include "TransitionMatrix.h"

TransitionMatrix::TransitionMatrix() {
    clear();
}

TransitionMatrix::TransitionMatrix(const TransitionMatrix& orig) {
    clear();
}

TransitionMatrix::~TransitionMatrix() {
}

void TransitionMatrix::clear(){
    stateStart.assign(1,0);
    pairStart.assign(1,0);
    column.clear();
    prob.clear();
    fillPosition.clear();
    shapeMismatches.clear();
    columnRows = -1;
}

void TransitionMatrix::fillTables(CSRTables& tables) const{
    tables.numberOfStates = numberOfRows();
    tables.stateStart = stateStart.data();
    tables.pairStart = pairStart.data();
    tables.column = column.data();
    tables.prob = prob.data();
}

void TransitionMatrix::reserve(int64_t nPairs, int64_t nEntries){
    pairStart.reserve(nPairs + 1);
    column.reserve(nEntries);
    prob.reserve(nEntries);
}

void TransitionMatrix::shrink(){
    column.shrink_to_fit();
    prob.shrink_to_fit();
    pairStart.shrink_to_fit();
    stateStart.shrink_to_fit();
}

void TransitionMatrix::setActions(const vector<int>& nActions){
    clear();
    stateStart.resize(nActions.size() + 1);
    for (size_t sidx=0; sidx<nActions.size(); sidx++){
        stateStart[sidx+1] = stateStart[sidx] + nActions[sidx];
    }
    pairStart.assign(stateStart.back() + 1, 0);
}

void TransitionMatrix::allocateEntries(){
    //pairStart holds the count of pair i in entry i+1: sum them up into offsets
    for (size_t i=1; i<pairStart.size(); i++){
        pairStart[i] += pairStart[i-1];
    }
    column.assign(pairStart.back(), -1);
    prob.assign(pairStart.back(), 0.0);
    fillPosition.assign(pairStart.begin(), pairStart.end() - 1);
}
