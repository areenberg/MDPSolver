#ifndef MODELTYPE_H
#define MODELTYPE_H

#include <cstdint>

//The transitions and rewards of a model, stored as tables in compressed sparse row (CSR)
//form. The state-action pairs are numbered state by state, and within a state in the order
//of its actions, so that pair stateStart[s] + a is action a of state s.
struct CSRTables {
    int numberOfStates = 0;
    const int64_t* stateStart = nullptr; //numberOfStates+1 entries: the pairs of state s are stateStart[s], ..., stateStart[s+1]-1
    const int64_t* pairStart = nullptr;  //one entry more than there are pairs: the non-zero transition probabilities of pair i are pairStart[i], ..., pairStart[i+1]-1
    const int* column = nullptr;         //the next state of each non-zero transition probability
    const double* prob = nullptr;        //and the probability itself, in double precision,
    const float* prob32 = nullptr;       //or in single precision (float32): exactly one of the two is set
    const double* reward = nullptr;      //the reward of each pair
};

//A model the solver can solve. There are two ways to supply one:
// - Store the transitions and rewards as CSR tables and return them from tables(), as the
//   general MDP model does. The solver then reads the arrays directly in its inner loops,
//   which is much faster than calling the methods below for every transition.
// - Compute the transitions on the fly in the methods below, as the built-in TBM and CBM
//   models do, and leave tables() returning nullptr. The solver then runs its routines for
//   such models, which call these methods. This suits models whose transitions follow a
//   formula and would take too much memory to store.
//(A special-case model of the second kind can also fill CSR tables once from its formulas
//and return them, if they fit in memory, to be solved with the fast loops.)
class ModelType {
public:

    virtual ~ModelType() {}

    //the model's CSR tables, or nullptr if it computes its transitions on the fly
    virtual const CSRTables* tables() { return nullptr; }

    virtual double getDiscount() = 0;
    virtual int getNumberOfStates() = 0;
    virtual void updateNumberOfActions(int &sidx) = 0;
    virtual int getNumberOfActions() = 0;
    virtual int getNumberOfActions(int &sidx) = 0;
    virtual int * getNextState() = 0;
    virtual double getPsj() = 0;
    virtual double reward(int &sidx, int &aidx) = 0;
    virtual double transProb(int &sidx, int &aidx, int &jidx) = 0;
    virtual void updateNextState(int &sidx, int &aidx, int &jidx) = 0;
    virtual int postDecisionIdx(int &sidx, int &aidx) = 0;
    virtual int getColumnIdx(int &sidx, int &aidx, int &cidx) = 0;
    virtual int getNumberOfJumps(int &sidx, int &aidx) = 0;

};

#endif  // MODELTYPE_H
