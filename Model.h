/*
 * File:   Model.h
 * Author: Jesper Fink Andersen
 *
 * Created on 1. october 2020, 17:00
 */

//Base class for MDP objects

#ifndef MODEL_H
#define MODEL_H

#include <vector>

class Model {
public:
    //general MDP parameters
    double discount;
    int numberOfStates;
    int numberOfActions;
    vector<int> policy;

    //auxiliary variables
    int sNext; //next state to process
    double pNext; //transition probability to sNext

    //functions
    virtual double reward(int, int);
    virtual double transProb(int, int, int);
    virtual void updateNext(int, int, int);
    virtual int sFirst(int, int);
};

#endif /* MODEL_H */