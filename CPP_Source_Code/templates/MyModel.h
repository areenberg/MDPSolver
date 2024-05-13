//Template header-file for the built-in MPD models.


#ifndef MYMODEL_H
#define MYMODEL_H

#include "ModelType.h"
#include "Policy.h"
#include <vector>

using namespace std;

class MyModel : public ModelType{
public:
    
    //constructor and destructor
    MyModel() {}; //dummy-constructor
    MyModel(double discount=0.99,int var1=2,int var2=10);
    MyModel(const MyModel& orig);
    virtual ~MyModel();

    //PUBLIC VARIABLES
    
    //add your variables here


    //PUBLIC METHODS

    //add your methods here


    //MANDATORY VARIABLES
    double discount,psj;
    int numberOfStates,numberOfActions,nextState;
    
    //MANDATORY METHODS (DO NOT CHANGE)   
    double reward(int &sidx, int &aidx) override;
    double transProb(int &sidx, int &aidx, int &jidx) override;
    void updateNextState(int &sidx, int &aidx, int &jidx) override;
    int postDecisionIdx(int &sidx, int &aidx) override;
    double getDiscount() override;
    int getNumberOfStates() override;
    void updateNumberOfActions(int &sidx) override;
    int getNumberOfActions() override;
    int getNumberOfActions(int &sidx) override;
    int * getNextState() override;
    double getPsj() override;
    int getNumberOfJumps(int &sidx, int &aidx) override;
    int getColumnIdx(int &sidx, int &aidx, int &cidx) override;
    
private:

    //VARIABLES

    //add your variables here
    

    //METHODS

    //add your methods here
    

};

#endif /* MYMODEL_H */
