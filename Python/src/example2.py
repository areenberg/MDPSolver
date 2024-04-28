import mdpsolver

#EXAMPLE 2
#In this example, we load the rewards and transition probabilities
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
# VALUE ITERATION
#---------------------------------------

mdl1 = mdpsolver.model()

mdl1.mdp(discount=0.99,
        rewardsFromFile="example_parameters/rewards_example2.csv",
        tranMatFromFile="example_parameters/transitions.csv")

mdl1.solve(algorithm="vi")

print("Value iteration:",mdl1.getRuntime(),"milliseconds.")

#---------------------------------------
# POLICY ITERATION (GAUSS-SEIDEL UPDATES)
#---------------------------------------

mdl2 = mdpsolver.model()

mdl2.mdp(discount=0.99,
        rewardsFromFile="example_parameters/rewards_example2.csv",
        tranMatFromFile="example_parameters/transitions.csv")

mdl2.solve(algorithm="pi",update="gs")

print("Policy iteration (w. Gauss-Seidel updates):",mdl2.getRuntime(),"milliseconds.")

#---------------------------------------
# MODIFIED POLICY ITERATION
#---------------------------------------

mdl3 = mdpsolver.model()

mdl3.mdp(discount=0.99,
        rewardsFromFile="example_parameters/rewards_example2.csv",
        tranMatFromFile="example_parameters/transitions.csv")

mdl3.solve(algorithm="mpi")

print("Modified policy iteration:",mdl3.getRuntime(),"milliseconds.")
