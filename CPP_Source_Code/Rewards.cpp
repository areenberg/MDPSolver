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

#include "Rewards.h"
#include <limits>

Rewards::Rewards() {
    clear();
}

Rewards::Rewards(const Rewards& orig) {
    clear();
}

Rewards::~Rewards() {
}

void Rewards::clear(){
    stateStart.assign(1,0);
    values.clear();
}

void Rewards::setActions(const vector<int>& nActions){
    stateStart.resize(nActions.size() + 1);
    stateStart[0] = 0;
    for (size_t sidx=0; sidx<nActions.size(); sidx++){
        stateStart[sidx+1] = stateStart[sidx] + nActions[sidx];
    }
    //rewards that the input leaves out stay NaN, which checkModel() reports
    values.assign(stateStart.back(), numeric_limits<double>::quiet_NaN());
}
