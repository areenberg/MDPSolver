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

using namespace std;

#ifndef REWARDS_H
#define REWARDS_H

//The rewards of the general MDP model, one per state-action pair, in one flat array that is
//numbered like the pairs of TransitionMatrix: state by state, and within a state in the order
//of its actions. When the rewards have the same numbers of actions as the transition
//probabilities, which is the case in a well-formed model, the solver reads this array as it
//is (see GeneralMDPmodel).
class Rewards {
public:

    Rewards();
    Rewards(const Rewards& orig);
    virtual ~Rewards();

    //METHODS

    //sizes
    int numberOfRows() const { return (int)stateStart.size() - 1; } //the number of states
    int numberOfActions(int sidx) const { return (int)(stateStart[sidx+1] - stateStart[sidx]); }

    //read and write values
    double getReward(int sidx, int aidx) const { return values[stateStart[sidx] + aidx]; }
    void assignReward(double reward, int sidx, int aidx) { values[stateStart[sidx] + aidx] = reward; }
    const double* data() const { return values.data(); }

    //BUILDING (by the loaders in ModuleInterface)

    void clear(); //no states

    //in order: the rewards of each state, then the states
    void reserve(int64_t nPairs) { values.reserve(nPairs); }
    void addReward(double reward) { values.push_back(reward); }
    void endState() { stateStart.push_back((int64_t)values.size()); }

    //or the number of actions of each state first, then the rewards in any order
    //(rewards that the input leaves out stay NaN, which checkModel() reports)
    void setActions(const vector<int>& nActions);

private:

    //VARIABLES
    vector<int64_t> stateStart; //numberOfRows()+1 entries: the rewards of state s are stateStart[s], ..., stateStart[s+1]-1
    vector<double> values; //the reward of each state-action pair

};

#endif /* REWARDS_H */

