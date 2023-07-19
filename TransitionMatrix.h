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


using namespace std;

#ifndef TRANSITIONMATRIX_H
#define TRANSITIONMATRIX_H

class TransitionMatrix {
public:
    
    TransitionMatrix();
    TransitionMatrix(const TransitionMatrix& orig);
    virtual ~TransitionMatrix();
    
    //METHODS
    
    //read and write values
    inline double* getProb(int& sidx, int& aidx, int& jidx);
    inline int* getColumn(int& sidx, int& aidx, int& jidx);
    void assignProb(double prob, int& sidx, int& aidx, int& jidx);
    void assignColumn(int column, int& sidx, int& aidx, int& jidx);
    
    //set size of matrix
    void setNumberOfRows(int numberOfStates);
    void setNumberOfActions(int nActions, int& sidx);
    void setNumberOfColumns(int nJumps, int& sidx, int& aidx);
    
    inline size_t numberOfColumns(int& sidx, int& aidx);
    inline size_t numberOfActions(int& sidx);
    inline size_t numberOfRows();
    
private:

    //VARIABLES
    vector<vector<vector<double>>> probs; //non-zero probabilities in the transition matrix
    vector<vector<vector<int>>> cols; //corresponding column indices in the transition matrix
    
};

#endif /* TRANSITIONMATRIX_H */

