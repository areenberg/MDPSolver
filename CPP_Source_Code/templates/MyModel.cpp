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
	return 0;
}

double MyModel::transProb(int &sidx, int &aidx, int &jidx) {
    return 0;
}

void MyModel::updateNextState(int &sidx, int &aidx, int &jidx) {
}

int MyModel::postDecisionIdx(int &sidx, int &aidx) {
    return 0;
}

double MyModel::getDiscount(){
    return discount;
}

int MyModel::getNumberOfStates(){
    return numberOfStates;
}

void MyModel::updateNumberOfActions(int &sidx){	
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
	return 0;
}

int MyModel::getNumberOfJumps(int &sidx, int &aidx){
	return 0;
}

int MyModel::getNumberOfActions(int &sidx){
	return 0;
}