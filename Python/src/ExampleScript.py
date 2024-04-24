import mdpsolver

#Example with 3 states and 2 actions in each state.

rewards = [[5,-1],
           [1,-2],
           [50,0]]

#alternative representation [stateFrom,action,reward]
#rewardsElementwise = [[0,0,0.5],
#                      [0,1,-0.1],
#                      [1,0,1.0],
#                      [1,1,-0.2],
#                      [2,0,0.0],
#                      [2,1,50.0]]

tranMatWithZeros = [[[0.9,0.1,0.0],[0.1,0.9,0.0]],
                    [[0.4,0.5,0.1],[0.3,0.5,0.2]],
                    [[0.2,0.2,0.6],[0.5,0.5,0.0]]]

#elementwise representation [stateFrom,action,stateTo,probability]
#tranMatElementwise = [[0,0,0,0.9],
#                      [0,0,1,0.1],
#                      [0,1,0,0.1],
#                      [0,1,1,0.9],
#                      [1,0,0,0.4],
#                      [1,0,1,0.5],
#                      [1,0,2,0.1],
#                      [1,1,0,0.3],
#                      [1,1,1,0.5],
#                      [1,1,2,0.2],
#                      [2,0,0,0.2],
#                      [2,0,1,0.2],
#                      [2,0,2,0.6],
#                      [2,1,0,0.5],
#                      [2,1,1,0.5]]

#probabilities and columns in separate lists
#tranMatProbs = [[[0.9,0.1],[0.1,0.9]],
#                [[0.4,0.5,0.1],[0.3,0.5,0.2]],
#                [[0.2,0.2,0.6],[0.5,0.5]]]

#tranMatColumns = [[[0,1],[0,1]],
#                [[0,1,2],[0,1,2]],
#                [[0,1,2],[0,1]]]


mdl = mdpsolver.model()
mdl.mdp(discount=0.8,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)

#alternative representation
#mdl.mdp(discount=0.8,
#        rewards=rewards,
#        tranMatElementwise=tranMatElementwise)

mdl.solve()

print(mdl.getPolicy())


#print(mdl.getValueVector())
#mdl.saveToFile()

#for i in range(3):
#        print(mdl.getAction(i),mdl.getValue(i))

#mdl.saveToFile("optimal_policy.csv")
#mdl.saveToFile("optimal_values.csv","values")




#SPECIFIC PROBLEMS

#mdl1 = mdpsolver.model()
#mdl1.tbm(discount=0.99,components=2,stages=10)
#mdl1.solve()

#print(mdl1.getPolicy())
#print(mdl1.aux.actionToReplace(mdl1.getAction(mdl1.aux.compToState([9,9]))))

#mdl2 = mdpsolver.model()
#mat = [[0.0,0.1,0.2,0.4,0.2,0.1,0.00,0.00,0.00,0.0],
#       [0.0,0.1,0.2,0.4,0.2,0.1,0.00,0.00,0.00,0.0]]

#mdl2.cbm(discount=0.99,components=2,stages=10,pCompMat=mat)
#mdl2.solve()

#print(mdl2.getPolicy())
#print(mdl2.aux.actionToReplace(mdl2.getAction(mdl2.aux.compToState([9,9]))))
