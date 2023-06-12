#ifndef MODELTYPE_H
#define MODELTYPE_H

class ModelType {
public:
    
    virtual int getPolicy(int) = 0;
    virtual void assignPolicy(int,int) = 0;
    virtual double getDiscount() = 0;
    virtual int getNumberOfStates() = 0;
    virtual int getNumberOfActions() = 0;
    virtual int getNextState() = 0;
    virtual double getPsj() = 0;
    virtual double reward(int, int) = 0;
    virtual double transProb(int, int, int) = 0;
    virtual void updateNextState(int, int, int) = 0;
    virtual int postDecisionIdx(int, int) = 0;
    
};

#endif  // MODELTYPE_H