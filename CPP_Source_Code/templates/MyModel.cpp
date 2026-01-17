//Template CPP-file for the built-in MPD models.

#include "MyModel.h"

using namespace std;

MyModel::MyModel(double discount=0.99,int var1=2,int var2=10):
	discount(discount)	
{
    //add constructor content here
}

MyModel::MyModel(const MyModel& orig) {
}

MyModel::~MyModel() {
}

double MyModel::reward(int &sidx,int &aidx) {
    //calculates and returns the reward of taking action
    //aidx when in current state sidx

	return 0;
}

double MyModel::transProb(int &sidx, int &aidx, int &jidx) {
    //calculates the probability of jumping to state jidx from
    //the current state sidx when taking action aidx.
    //returns the calculated probability.

    return 0;
}

void MyModel::updateNextState(int &sidx, int &aidx, int &jidx) {
    //updates the next possible state, nextState, and the associated
    //transition probability, psj (see e.g. GeneralMDPmodel.cpp).

}

int MyModel::postDecisionIdx(int &sidx, int &aidx) {
    //derives the first new/next state that is possible
    //to reach from the current state sidx.
    //both returns the value of the first new state, and
    //stores it in the variable nextState.

    return 0;    
}

double MyModel::getDiscount(){
    return discount;
}

int MyModel::getNumberOfStates(){
    //computes and/or returns the number of 
    //states (i.e. size of the state space).

    return 0;
}

void MyModel::updateNumberOfActions(int &sidx){
    //computes the number of actions (i.e. size of the action space)
    //associated with currnent state sidx and stores the result
    //in the variable `numberOfActions`.

}

int MyModel::getNumberOfActions(){
    return numberOfActions;
}

int * MyModel::getNextState(){
    return &nextState;
}

double MyModel::getPsj(){
    return psj;
}

int MyModel::getColumnIdx(int &sidx, int &aidx, int &cidx){
    //returns the `nextState` (i.e. jidx) associated with
    //the current state sidx, action aidx, and index of
    //non-zero transition cidx. 
	return 0;
}

int MyModel::getNumberOfJumps(int &sidx, int &aidx){
    //computes and returns the possible number of
    //jumps from the current state sidx when taking
    //action aidx.

	return 0;
}

int MyModel::getNumberOfActions(int &sidx){
    //computes and returns the number of actions
    //(i.e. the action space size) associated with
    //current state sidx.
    
	return 0;
}