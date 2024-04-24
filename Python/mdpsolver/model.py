from mdpsolver import solvermodule
from mdpsolver.tbmaid import tbmaid
from mdpsolver.cbmaid import cbmaid

class model:
    
    def __init__(self):
        self.initialize()            
        
    def initialize(self):
        #create the solver object
        self.mdl = solvermodule.Model()
    
    def solve(self,
              algorithm="mpi",
              tolerance=1e-3,
              update="standard",
              parIterLim=100,
              SORrelaxation=1.0,
              verbose=False):
        #solves the problem employing the specified algorithm
        self.mdl.solve(algorithm=algorithm,
              tolerance=tolerance,
              update=update,
              parIterLim=parIterLim,
              SORrelaxation=SORrelaxation,
              verbose=verbose)
    
    def printPolicy(self):
        #print the entire policy in the terminal
        self.mdl.printPolicy()
    
    def printValueVector(self):
        #print the entire value vector in the terminal
        self.mdl.printValueVector()
    
    def getAction(self,stateIndex=0):
        #returns an action from the optimized policy
        return self.mdl.getAction(stateIndex=stateIndex)
    
    def getValue(self,stateIndex=0):
        #returns a value from the optimized value vector
        return self.mdl.getValue(stateIndex=stateIndex)
    
    def getPolicy(self):
        #returns the entire policy
        return self.mdl.getPolicy()
    
    def getValueVector(self):
        #returns the entire value vector
        return self.mdl.getValueVector()
    
    def saveToFile(self,fileName="result.csv",type="policy"):
        #saves the optimized policy or value vector to a comma-separated file 
        return self.mdl.saveToFile(fileName=fileName,type=type)
    
    def mdp(self,
            discount=0.99,
            rewards=list(),
            rewardsElementwise=list(),
            rewardsFromFile="rewards.csv",
            tranMatWithZeros=list(),
            tranMatElementwise=list(),
            tranMatProbs=list(),
            tranMatColumns=list(),
            tranMatFromFile="transitions.csv"
            ):
        #the generic MDP model
        self.mdl.mdp(discount=discount,
            rewards=rewards,
            rewardsElementwise=rewardsElementwise,
            rewardsFromFile=rewardsFromFile,
            tranMatWithZeros=tranMatWithZeros,
            tranMatElementwise=tranMatElementwise,
            tranMatProbs=tranMatProbs,
            tranMatColumns=tranMatColumns,
            tranMatFromFile=tranMatFromFile)

    def tbm(self,discount=0.99,components=2,stages=10):
        #the time-based maintenance problem
        self.mdl.tbm(discount=discount,components=components,stages=stages)
        self.aid = tbmaid(components=components,stages=stages)
    
    def cbm(self,discount=0.99,components=2,stages=10,pCompMat=list()):
        #the condition-based maintenance problem
        self.mdl.cbm(discount=discount,components=components,stages=stages,pCompMat=pCompMat)
        self.aid = cbmaid(components=components,stages=stages)