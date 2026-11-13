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

#include <vector>
#include <cstdint>
#include "ModelType.h"

using namespace std;

#ifndef TRANSITIONMATRIX_H
#define TRANSITIONMATRIX_H

//The non-zero transition probabilities of the general MDP model, in compressed sparse row
//(CSR) form: four flat arrays instead of nested vectors, so that the solver can run through
//them directly (see CSRTables in ModelType.h). The state-action pairs are numbered state by
//state, and within a state in the order of its actions; the non-zero probabilities of each
//pair are kept in the order the input gave them, so that sums over them come out the same
//as before the change to CSR. The probabilities are kept in double precision, or in single
//precision (float32) when the input gave them so: 8 instead of 12 bytes per non-zero, and
//since a float32 number converts to double exactly, the same results as the same numbers
//in double precision.
class TransitionMatrix {
public:

    TransitionMatrix();
    TransitionMatrix(const TransitionMatrix& orig);
    virtual ~TransitionMatrix();

    //METHODS

    //sizes
    int numberOfRows() const { return (int)stateStart.size() - 1; } //the number of states
    int numberOfActions(int sidx) const { return (int)(stateStart[sidx+1] - stateStart[sidx]); }
    int numberOfColumns(int sidx, int aidx) const { //the number of non-zero probabilities of a state-action pair
        int64_t i = stateStart[sidx] + aidx;
        return (int)(pairStart[i+1] - pairStart[i]);
    }

    //read values (cidx counts the non-zero probabilities of the pair)
    double getProb(int sidx, int aidx, int cidx) const {
        int64_t k = pairStart[stateStart[sidx] + aidx] + cidx;
        return single ? (double)prob32[k] : prob[k];
    }
    int getColumn(int sidx, int aidx, int cidx) const { return column[pairStart[stateStart[sidx] + aidx] + cidx]; }

    //points the arrays of the tables at this matrix (the rewards are the model's)
    void fillTables(CSRTables& tables) const;

    //BUILDING (by the loaders in ModuleInterface)

    void clear(); //no states

    //whether the probabilities are kept in single precision (float32); set it after clear()
    //or setActions() and before the entries
    void setSinglePrecision(bool singlePrecision) { single = singlePrecision; }
    bool singlePrecision() const { return single; }

    //in order: the entries of each pair, the pairs of each state, the states
    void reserve(int64_t nPairs, int64_t nEntries);
    void addEntry(int col, double p) {
        column.push_back(col);
        if (single){
            prob32.push_back((float)p);
        }else{
            prob.push_back(p);
        }
    }
    void endPair() { pairStart.push_back((int64_t)column.size()); }
    void endState() { stateStart.push_back((int64_t)pairStart.size() - 1); }
    void shrink(); //frees the capacity that addEntry() left unused

    //or in any order, in three steps: the number of actions of each state; one count per
    //entry; then the entries, which keep the order they are filled in within each pair
    void setActions(const vector<int>& nActions);
    void countEntry(int sidx, int aidx) { pairStart[stateStart[sidx] + aidx + 1]++; }
    void allocateEntries();
    void fillEntry(int sidx, int aidx, int col, double p) {
        int64_t k = fillPosition[stateStart[sidx] + aidx]++;
        column[k] = col;
        if (single){
            prob32[k] = (float)p;
        }else{
            prob[k] = p;
        }
    }
    void finishEntries() { vector<int64_t>().swap(fillPosition); }

    //HOW tranMatProbs AND tranMatColumns DIFFER IN SHAPE (for checkModel())
    //When the two lists differ, a pair keeps as many entries as both have, and the
    //actions and states that tranMatColumns lacks have no entries. The differences are
    //recorded while loading, in the order checkModel() reports them.
    struct ShapeMismatch {
        int sidx;
        int aidx; //-1: the numbers of actions of state sidx differ
        int nProbs; //entries (or actions) in tranMatProbs
        int nColumns; //and in tranMatColumns
    };
    vector<ShapeMismatch> shapeMismatches;
    int columnRows = -1; //the number of states in tranMatColumns (-1: the format has no separate list of columns)
    int numberOfColumnRows() const { return (columnRows < 0) ? numberOfRows() : columnRows; }

private:

    //VARIABLES
    vector<int64_t> stateStart; //numberOfRows()+1 entries: the pairs of state s are stateStart[s], ..., stateStart[s+1]-1
    vector<int64_t> pairStart; //one entry more than there are pairs: the non-zero probabilities of pair i are pairStart[i], ..., pairStart[i+1]-1
    vector<int> column; //the next state (column of the transition matrix) of each non-zero probability
    vector<double> prob; //and the probability, in double precision,
    vector<float> prob32; //or in single precision (only one of the two is used)
    bool single = false; //whether prob32 is the one
    vector<int64_t> fillPosition; //while filling in any order: the next free entry of each pair

};

#endif /* TRANSITIONMATRIX_H */

