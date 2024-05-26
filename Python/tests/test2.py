import mdpsolver
import example_model

#TEST 2
#In this test, we load the rewards and transition probabilities
#from two CSV-files (rewards_example.csv and transitions.csv).
#In addition, we test three optimization algorithms and compare
#the runtimes.

#PROBLEM DESCRIPTION: We derive the policy for a decision problem
#involving a queueing system with finite capacity and two customer
#types. The system is observed whenever a new customer arrives.
#Customers are not allowed to wait and can only be accepted if there
#are idle servers. Thus, with each arrival, we must choose whether
#to accept or reject the customer. We receive a low reward for
#accepting the first, more common type, and a high reward for
#accepting the second, rarer type.

#---------------------------------------
# CREATE PARAMETERS
#---------------------------------------

#rewards
capacity=100
rewardCust1=1
rewardCust2=1000
example_model.rewards(capacity,rewardCust1,rewardCust2,"rewards_example2.csv")

#transitions
arrRate1=50
arrRate2=10
serRate=3/4
example_model.tranMat(arrRate1,arrRate2,serRate,capacity,"transitions_example2.csv")

#---------------------------------------
# VALUE ITERATION
#---------------------------------------

mdl1 = mdpsolver.model()

mdl1.mdp(discount=0.99,
        rewardsFromFile="rewards_example2.csv",
        tranMatFromFile="transitions_example2.csv")

mdl1.solve(algorithm="vi")

print("Value iteration:",mdl1.getRuntime(),"milliseconds.")

#---------------------------------------
# POLICY ITERATION
#---------------------------------------

mdl2 = mdpsolver.model()

mdl2.mdp(discount=0.99,
        rewardsFromFile="rewards_example2.csv",
        tranMatFromFile="transitions_example2.csv")

mdl2.solve(algorithm="pi")

print("Policy iteration:",mdl2.getRuntime(),"milliseconds.")

#---------------------------------------
# MODIFIED POLICY ITERATION
#---------------------------------------

mdl3 = mdpsolver.model()

mdl3.mdp(discount=0.99,
        rewardsFromFile="rewards_example2.csv",
        tranMatFromFile="transitions_example2.csv")

mdl3.solve(algorithm="mpi")

print("Modified policy iteration:",mdl3.getRuntime(),"milliseconds.")