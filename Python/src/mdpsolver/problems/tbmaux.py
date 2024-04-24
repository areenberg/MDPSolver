import math

#Time-based maintenance problem

#Note: The class is only used for retrieving/converting information
#(i.e. returning an action of the policy). The policy is optimized 
#in the 'model' (model.py) class.

class tbmaux:

    def __init__(self,components,stages):
        self.components=components
        self.stages=stages
        
    def compToState(self,compStates):
        #input: list containing the stage of each component
        #output: the associated state index
        prod = self.components * [1]
        if self.components>1:
            for i in range(self.components-2,-1,-1):
                prod[i]=prod[i+1]*self.stages 
        stateIndex=0
        for i in range(self.components):
            stateIndex += compStates[i]*prod[i]    
        return stateIndex
    
    def actionToReplace(self,actionIndex):
        #input: action index from the optimized policy
        #output: list of bools indicating which comps to replace
        prod = self.components * [1]
        if self.components>1:
            for i in range(self.components-2,-1,-1):
                prod[i]=prod[i+1]*2
        replace = self.components * [False]        
        for i in range(self.components):
            if  math.floor(actionIndex/prod[i])>0:
                replace[i]=True
                actionIndex-=prod[i]             
        return replace