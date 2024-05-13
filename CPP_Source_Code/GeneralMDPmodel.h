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


#ifndef GENERALMDPMODEL_H
#define GENERALMDPMODEL_H

#include "ModelType.h"
#include "Policy.h"
#include "TransitionMatrix.h"
#include "Rewards.h"
#include <vector>

using namespace std;

class GeneralMDPmodel : public ModelType{
public:
    
    //CONSTRUCTOR AND DESCTRUCTOR
    GeneralMDPmodel() {}; //dummy-constructor
    GeneralMDPmodel(Rewards * rw, TransitionMatrix * tm, double discount);
    GeneralMDPmodel(const GeneralMDPmodel& orig);
    virtual ~GeneralMDPmodel();


    //VARIABLES

    double discount;
	int numberOfStates, numberOfActions;
    
    //auxiliary variables
    int nextState; //a new state that the current state can jump to
    double psj; //transition probability from the current state to nextState
	
    //METHODS
    void initialize();
        
    //GENERIC METHODS    
    double reward(int &sidx, int &aidx) override;
    double transProb(int &sidx, int &aidx, int &jidx) override;
    void updateNextState(int &sidx, int &aidx, int &jidx) override;
    int postDecisionIdx(int &sidx, int &aidx) override;
    int getNumberOfJumps(int &sidx, int &aidx) override;
    int getColumnIdx(int &sidx, int &aidx, int &cidx) override;
    double getDiscount() override;
    int getNumberOfStates() override;
    void updateNumberOfActions(int &sidx) override;
    int getNumberOfActions() override;
    int getNumberOfActions(int &sidx) override;
    int * getNextState() override;
    double getPsj() override;

private:

    //VARIABLES
    Rewards * rewards;
    TransitionMatrix * tranMat;
    int cidx;
    

};

#endif /* TBMMODEL_H */
