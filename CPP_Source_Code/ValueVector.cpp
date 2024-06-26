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


#include "ValueVector.h"

ValueVector::ValueVector() {
    initialize();
}

ValueVector::ValueVector(const ValueVector& orig) {
}

ValueVector::~ValueVector() {
}

void ValueVector::initialize(){
    valueVector.resize(1,-1);    
}

void ValueVector::setSize(int numberOfStates){
    valueVector.resize(numberOfStates,0.0);
}

void ValueVector::assignValue(int& sidx, double& value){
    valueVector[sidx] = value;
}

double* ValueVector::getValue(int& sidx){
    return &valueVector[sidx];
}