import mdpsolver
import example_model

#EXAMPLE 3
#In this example, we demonstrate how to select a custom
#initial policy and value vector.
 
#Specifically, we select the solution to the decision problem
#in Example 2 as the initial policy and value vector for the
#optimization of a similar problem with new rewards.

#---------------------------------------
# CREATE PARAMETERS
#---------------------------------------

#Old rewards (Example 2)
capacity=100
rewardCust1=1
rewardCust2=1000
example_model.rewards(capacity,rewardCust1,rewardCust2,"rewards_example2.csv")

#New rewards
rewardCust1=10
rewardCust2=980
example_model.rewards(capacity,rewardCust1,rewardCust2,"rewards_example3.csv")

#Same transitions (Example 2)
arrRate1=50
arrRate2=10
serRate=3/4
example_model.tranMat(arrRate1,arrRate2,serRate,capacity,"transitions_example2.csv")

#---------------------------------------
# SOLUTION TO ORIGINAL PROBLEM (EXAMPLE 2)
#---------------------------------------

mdl1 = mdpsolver.model()

mdl1.mdp(discount=0.99,
        rewardsFromFile="rewards_example2.csv",
        tranMatFromFile="transitions_example2.csv")

mdl1.solve(verbose=True,postProcessing=False)

#---------------------------------------
# SIMILAR PROBLEM - NEW REWARDS
#---------------------------------------

mdl2 = mdpsolver.model()

mdl2.mdp(discount=0.99,
        rewardsFromFile="rewards_example3.csv",
        tranMatFromFile="transitions_example2.csv")

mdl2.solve(initPolicy=mdl1.getPolicy(),initValueVector=mdl1.getValueVector(),verbose=True)