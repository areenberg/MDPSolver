import mdpsolver

#EXAMPLE 1
#Simple MDP with 3 states and 2 actions in each state.

#---------------------------------------
# PARAMETERS
#---------------------------------------

#rewards
#1st index: from (current) states
#2nd index: actions
rewards = [[5,-1],
           [1,-2],
           [50,0]]

#transition probabilities
#1st index: from (current) states
#2nd index: actions
#3rd index: to (next) states 
tranMatWithZeros = [[[0.9,0.1,0.0],[0.1,0.9,0.0]],
                    [[0.4,0.5,0.1],[0.3,0.5,0.2]],
                    [[0.2,0.2,0.6],[0.5,0.5,0.0]]]

#---------------------------------------
# MODEL
#---------------------------------------

#create the model object
mdl = mdpsolver.model()

#specify discount and insert parameters
mdl.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)

#---------------------------------------
# SOLVE
#---------------------------------------

#derive an epsilon-optimal policy for the
#MDP using the default algorithm settings
mdl.solve()

#---------------------------------------
# RESULTS
#---------------------------------------

#get the optimized policy and value
#vector

print(mdl.getPolicy())
#[1, 1, 0]

print(mdl.getValueVector())
#[200.00124208153733, 212.86682245402108, 298.709135275055]