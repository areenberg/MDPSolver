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

inline double* TransitionMatrix::getProb(int& sidx, int& aidx, int& jidx){
    return &probs[sidx][aidx][jidx];
}
 
inline int* TransitionMatrix::getColumn(int& sidx, int& aidx, int& jidx){
    return &cols[sidx][aidx][jidx];
}

void TransitionMatrix::assignProb(double prob, int& sidx, int& aidx, int& jidx){
    probs[sidx][aidx][jidx]=prob; 
}

void TransitionMatrix::assignColumn(int column, int& sidx, int& aidx, int& jidx){
    cols[sidx][aidx][jidx]=column;
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

inline size_t TransitionMatrix::numberOfColumns(int& sidx, int& aidx){
    return cols[sidx][aidx].size();
}

inline size_t TransitionMatrix::numberOfActions(int& sidx){
    return probs[sidx].size();
}

inline size_t TransitionMatrix::numberOfRows(){
    return probs.size();
}