import mdpsolver

#EXAMPLE 3
#In this example, we demonstrate how to select a custom
#initial policy and value vector.
 
#Specifically, we select the solution to the decision problem
#in Example 2 as the initial policy and value vector for the
#optimization of a similar problem with new rewards. 

#---------------------------------------
# SOLUTION TO ORIGINAL PROBLEM (EXAMPLE 2)
#---------------------------------------

mdl1 = mdpsolver.model()

mdl1.mdp(discount=0.99,
        rewardsFromFile="example_parameters/rewards_example2.csv",
        tranMatFromFile="example_parameters/transitions.csv")

mdl1.solve(verbose=True)

#---------------------------------------
# SIMILAR PROBLEM - NEW REWARDS
#---------------------------------------

mdl2 = mdpsolver.model()

mdl2.mdp(discount=0.99,
        rewardsFromFile="example_parameters/rewards_example3.csv",
        tranMatFromFile="example_parameters/transitions.csv")

mdl2.solve(initPolicy=mdl1.getPolicy(),initValueVector=mdl1.getValueVector(),verbose=True)