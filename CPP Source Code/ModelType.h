#ifndef MODELTYPE_H
#define MODELTYPE_H

class ModelType {
public:
    
    virtual double * getDiscount() = 0;
    virtual int * getNumberOfStates() = 0;
    virtual void updateNumberOfActions(int &sidx) = 0;
    virtual int * getNumberOfActions() = 0;
    virtual int * getNextState() = 0;
    virtual double * getPsj() = 0;
    virtual double * reward(int &sidx, int &aidx) = 0;
    virtual double * transProb(int &sidx, int &aidx, int &jidx) = 0;
    virtual void updateNextState(int &sidx, int &aidx, int &jidx) = 0;
    virtual int * postDecisionIdx(int &sidx, int &aidx) = 0;
    
};

#endif  // MODELTYPE_H