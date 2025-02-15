import mdpsolver
import random
import sys
import numpy as np
from random import randint

#TEST 1
#Simple MDP with 3 states and 2 actions in each state.

#---------------------------------------
# CONFIGURATION 1
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

#initial policy
random.seed(10)
initPolicy = [randint(0, 1) for p in range(0, 3)]

#Model 1a (discounted reward, parallel)
mdl1a = mdpsolver.model()
mdl1a.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1a.solve(algorithm="mpi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1a failed!")        
if not np.array_equal(np.round(np.array(mdl1a.getValueVector()),6),np.round(np.array([200.00114718191236,212.86672887958622,298.7090458676583]),6)):
        sys.exit("Model 1a failed!")

#Model 2a (discounted reward, parallel)
mdl2a = mdpsolver.model()
mdl2a.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2a.solve(algorithm="pi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2a failed!")        
if not np.array_equal(np.round(np.array(mdl2a.getValueVector()),6),np.round(np.array([200.00114718191236,212.86672887958622,298.7090458676583]),6)):
        sys.exit("Model 2a failed!")

#Model 3a (discounted reward, parallel)
mdl3a = mdpsolver.model()
mdl3a.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl3a.solve(algorithm="vi",
          update="standard",
          parallel=True)
if not np.array_equal(np.array(mdl3a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3a failed!")        
if not np.array_equal(np.round(np.array(mdl3a.getValueVector()),6),np.round(np.array([200.0013199959708, 212.86689665815788, 298.709197760198]),6)):
        sys.exit("Model 3a failed!")

#Model 1b (discounted reward, unparallel)
mdl1b = mdpsolver.model()
mdl1b.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1b.solve(algorithm="mpi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1b failed!")        
if not np.array_equal(np.round(np.array(mdl1b.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 1b failed!")

#Model 2b (discounted reward, unparallel)
mdl2b = mdpsolver.model()
mdl2b.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2b.solve(algorithm="pi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2b failed!")        
if not np.array_equal(np.round(np.array(mdl2b.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 2b failed!")

#Model 3b (discounted reward, unparallel)
mdl3b = mdpsolver.model()
mdl3b.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl3b.solve(algorithm="vi",
          update="standard",
          parallel=False)
if not np.array_equal(np.array(mdl3b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3b failed!")        
if not np.array_equal(np.round(np.array(mdl3b.getValueVector()),6),np.round(np.array([200.0013199959708, 212.86689665815788, 298.709197760198]),6)):
        sys.exit("Model 3b failed!")

#Model 1a (average reward, parallel)
mdl1a = mdpsolver.model()
mdl1a.mdp(rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1a.solve(algorithm="mpi",
          update="standard",
          criterion="average",
          parallel=True,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1a failed!")        
if not np.array_equal(np.round(np.array(mdl1a.getValueVector()),6),np.round(np.array([354.2101307157537, 368.21018917357264, 457.21037278800077]),6)):
        sys.exit("Model 1a failed!")

#Model 2a (average reward, parallel)
mdl2a = mdpsolver.model()
mdl2a.mdp(rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2a.solve(algorithm="pi",
          update="standard",
          criterion="average",
          parallel=True,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2a failed!")        
if not np.array_equal(np.round(np.array(mdl2a.getValueVector()),6),np.round(np.array([354.2101307157537, 368.21018917357264, 457.21037278800077]),6)):
        sys.exit("Model 2a failed!")

#Model 3a (average reward, parallel)
mdl3a = mdpsolver.model()
mdl3a.mdp(rewards=rewards,
          tranMatWithZeros=tranMatWithZeros)
mdl3a.solve(algorithm="vi",
          update="standard",
          criterion="average",
          parallel=True)
if not np.array_equal(np.array(mdl3a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3a failed!")        
if not np.array_equal(np.round(np.array(mdl3a.getValueVector()),6),np.round(np.array([158.23345351808334, 172.23340169038516, 261.233237936489]),6)):
        sys.exit("Model 3a failed!")

#Model 1b (average reward, unparallel)
mdl1b = mdpsolver.model()
mdl1b.mdp(rewards=rewards,
          tranMatWithZeros=tranMatWithZeros)
mdl1b.solve(algorithm="mpi",
          update="standard",
          criterion="average",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1b failed!")        
if not np.array_equal(np.round(np.array(mdl1b.getValueVector()),6),np.round(np.array([354.2101307157537, 368.21018917357264, 457.21037278800077]),6)):
        sys.exit("Model 1b failed!")

#Model 2b (average reward, unparallel)
mdl2b = mdpsolver.model()
mdl2b.mdp(rewards=rewards,
          tranMatWithZeros=tranMatWithZeros)
mdl2b.solve(algorithm="pi",
          update="standard",
          criterion="average",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2b failed!")        
if not np.array_equal(np.round(np.array(mdl2b.getValueVector()),6),np.round(np.array([354.2101307157537, 368.21018917357264, 457.21037278800077]),6)):
        sys.exit("Model 2b failed!")

#Model 3b (average reward, unparallel)
mdl3b = mdpsolver.model()
mdl3b.mdp(rewards=rewards,
          tranMatWithZeros=tranMatWithZeros)
mdl3b.solve(algorithm="vi",
          update="standard",
          criterion="average",
          parallel=False)
if not np.array_equal(np.array(mdl3b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3b failed!")        
if not np.array_equal(np.round(np.array(mdl3b.getValueVector()),6),np.round(np.array([158.23345351808334, 172.23340169038516, 261.233237936489]),6)):
        sys.exit("Model 3b failed!")

#Model 1c (discounted, Gauss-Seidel)
mdl1c = mdpsolver.model()
mdl1c.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1c.solve(algorithm="mpi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1c failed!")        
if not np.array_equal(np.round(np.array(mdl1c.getValueVector()),6),np.round(np.array([200.0012771893156, 212.86686671568802, 298.7091798650016]),6)):
        sys.exit("Model 1c failed!")

#Model 2c (Gauss-Seidel)
initPolicy=[1,1,1]
mdl2c = mdpsolver.model()
mdl2c.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2c.solve(algorithm="pi",
            update="gs",
            parallel=False,
            initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2c failed!")        
if not np.array_equal(np.round(np.array(mdl2c.getValueVector()),6),np.round(np.array([200.0012771893156, 212.86686671568802, 298.7091798650016]),6)):
        sys.exit("Model 2c failed!")

#Model 3c (Gauss-Seidel)
mdl3c = mdpsolver.model()
mdl3c.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl3c.solve(algorithm="vi",
          update="gs",
          parallel=False)
if not np.array_equal(np.array(mdl3c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3c failed!")        
if not np.array_equal(np.round(np.array(mdl3c.getValueVector()),6),np.round(np.array([200.0010604606014, 212.86664512773723, 298.7089561808565]),6)):
        sys.exit("Model 3c failed!")

#Model 1d (SOR)
mdl1d = mdpsolver.model()
mdl1d.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1d.solve(algorithm="mpi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1d failed!")        
if not np.array_equal(np.round(np.array(mdl1d.getValueVector()),6),np.round(np.array([200.00127021272166, 212.86686059325598, 298.70917425463125]),6)):
        sys.exit("Model 1d failed!")

#Model 2d (SOR)
mdl2d = mdpsolver.model()
mdl2d.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2d.solve(algorithm="pi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2d failed!")        
if not np.array_equal(np.round(np.array(mdl2d.getValueVector()),6),np.round(np.array([200.00127021272166, 212.86686059325598, 298.70917425463125]),6)):
        sys.exit("Model 2d failed!")

#Model 3d (SOR)
mdl3d = mdpsolver.model()
mdl3d.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl3d.solve(algorithm="vi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False)
if not np.array_equal(np.array(mdl3d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3d failed!")        
if not np.array_equal(np.round(np.array(mdl3d.getValueVector()),6),np.round(np.array([206.69740452375487, 222.30968454932525, 305.91165352015094]),6)):
        sys.exit("Model 3d failed!")

#---------------------------------------
# CONFIGURATION 2
#---------------------------------------

#rewards
#1st index: from (current) states
#2nd index: actions
rewards = [[5,-1],
           [1,-2],
           [50,0]]

#transition probabilities
#[from_state,action,to_state,probability]
tranMatElementwise = [[0,0,0,0.9],
                      [0,0,1,0.1],
                      [0,1,0,0.1],
                      [0,1,1,0.9],
                      [1,0,0,0.4],
                      [1,0,1,0.5],
                      [1,0,2,0.1],
                      [1,1,0,0.3],
                      [1,1,1,0.5],
                      [1,1,2,0.2],
                      [2,0,0,0.2],
                      [2,0,1,0.2],
                      [2,0,2,0.6],
                      [2,1,0,0.5],
                      [2,1,1,0.5]]

#initial policy
random.seed(10)
initPolicy = [randint(0, 1) for p in range(0, 3)]

#Model 1a
mdl1a = mdpsolver.model()
mdl1a.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl1a.solve(algorithm="mpi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1a failed!")        
if not np.array_equal(np.round(np.array(mdl1a.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 1a failed!")

#Model 2a
mdl2a = mdpsolver.model()
mdl2a.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl2a.solve(algorithm="pi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2a failed!")        
if not np.array_equal(np.round(np.array(mdl2a.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 2a failed!")

#Model 3a
mdl3a = mdpsolver.model()
mdl3a.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl3a.solve(algorithm="vi",
          update="standard",
          parallel=True)
if not np.array_equal(np.array(mdl3a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3a failed!")        
if not np.array_equal(np.round(np.array(mdl3a.getValueVector()),6),np.round(np.array([200.0013199959708, 212.86689665815788, 298.709197760198]),6)):
        sys.exit("Model 3a failed!")

#Model 1b
mdl1b = mdpsolver.model()
mdl1b.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl1b.solve(algorithm="mpi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1b failed!")        
if not np.array_equal(np.round(np.array(mdl1b.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 1b failed!")

#Model 2b
mdl2b = mdpsolver.model()
mdl2b.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl2b.solve(algorithm="pi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2b failed!")        
if not np.array_equal(np.round(np.array(mdl2b.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 2b failed!")

#Model 3b
mdl3b = mdpsolver.model()
mdl3b.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl3b.solve(algorithm="vi",
          update="standard",
          parallel=False)
if not np.array_equal(np.array(mdl3b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3b failed!")        
if not np.array_equal(np.round(np.array(mdl3b.getValueVector()),6),np.round(np.array([200.0013199959708, 212.86689665815788, 298.709197760198]),6)):
        sys.exit("Model 3b failed!")

#Model 1c
mdl1c = mdpsolver.model()
mdl1c.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl1c.solve(algorithm="mpi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1c failed!")        
if not np.array_equal(np.round(np.array(mdl1c.getValueVector()),6),np.round(np.array([200.0012771893156, 212.86686671568802, 298.7091798650016]),6)):
        sys.exit("Model 1c failed!")

#Model 2c
mdl2c = mdpsolver.model()
mdl2c.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl2c.solve(algorithm="pi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2c failed!")        
if not np.array_equal(np.round(np.array(mdl2c.getValueVector()),6),np.round(np.array([200.0012771893156, 212.86686671568802, 298.7091798650016]),6)):
        sys.exit("Model 2c failed!")

#Model 3c
mdl3c = mdpsolver.model()
mdl3c.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl3c.solve(algorithm="vi",
          update="gs",
          parallel=False)
if not np.array_equal(np.array(mdl3c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3c failed!")        
if not np.array_equal(np.round(np.array(mdl3c.getValueVector()),6),np.round(np.array([200.0010604606014, 212.86664512773723, 298.7089561808565]),6)):
        sys.exit("Model 3c failed!")

#Model 1d
mdl1d = mdpsolver.model()
mdl1d.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl1d.solve(algorithm="mpi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1d failed!")        
if not np.array_equal(np.round(np.array(mdl1d.getValueVector()),6),np.round(np.array([200.00127021272166, 212.86686059325598, 298.70917425463125]),6)):
        sys.exit("Model 1d failed!")

#Model 2d
mdl2d = mdpsolver.model()
mdl2d.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl2d.solve(algorithm="pi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2d failed!")        
if not np.array_equal(np.round(np.array(mdl2d.getValueVector()),6),np.round(np.array([200.00127021272166, 212.86686059325598, 298.70917425463125]),6)):
        sys.exit("Model 2d failed!")

#Model 3d
mdl3d = mdpsolver.model()
mdl3d.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl3d.solve(algorithm="vi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False)
if not np.array_equal(np.array(mdl3d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3d failed!")        
if not np.array_equal(np.round(np.array(mdl3d.getValueVector()),6),np.round(np.array([206.69740452375487, 222.30968454932525, 305.91165352015094]),6)):
        sys.exit("Model 3d failed!")

#---------------------------------------
# CONFIGURATION 3
#---------------------------------------

#rewards
#1st index: from (current) states
#2nd index: actions
rewards = [[5,-1],
           [1,-2],
           [50,0]]

#transition probabilities
tranMatProbs = [[[0.9,0.1],[0.1,0.9]],
                [[0.4,0.5,0.1],[0.3,0.5,0.2]],
                [[0.2,0.2,0.6],[0.5,0.5]]]

tranMatColumns = [[[0,1],[0,1]],
                [[0,1,2],[0,1,2]],
                [[0,1,2],[0,1]]]

#initial policy
random.seed(10)
initPolicy = [randint(0, 1) for p in range(0, 3)]

#Model 1a
mdl1a = mdpsolver.model()
mdl1a.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl1a.solve(algorithm="mpi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1a failed!")        
if not np.array_equal(np.round(np.array(mdl1a.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 1a failed!")

#Model 2a
mdl2a = mdpsolver.model()
mdl2a.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl2a.solve(algorithm="pi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2a failed!")        
if not np.array_equal(np.round(np.array(mdl2a.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 2a failed!")

#Model 3a
mdl3a = mdpsolver.model()
mdl3a.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl3a.solve(algorithm="vi",
          update="standard",
          parallel=True)
if not np.array_equal(np.array(mdl3a.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3a failed!")        
if not np.array_equal(np.round(np.array(mdl3a.getValueVector()),6),np.round(np.array([200.0013199959708, 212.86689665815788, 298.709197760198]),6)):
        sys.exit("Model 3a failed!")

#Model 1b
mdl1b = mdpsolver.model()
mdl1b.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl1b.solve(algorithm="mpi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1b failed!")        
if not np.array_equal(np.round(np.array(mdl1b.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 1b failed!")

#Model 2b
mdl2b = mdpsolver.model()
mdl2b.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl2b.solve(algorithm="pi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2b failed!")        
if not np.array_equal(np.round(np.array(mdl2b.getValueVector()),6),np.round(np.array([200.00114718191236, 212.86672887958622, 298.7090458676583]),6)):
        sys.exit("Model 2b failed!")

#Model 3b
mdl3b = mdpsolver.model()
mdl3b.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl3b.solve(algorithm="vi",
          update="standard",
          parallel=False)
if not np.array_equal(np.array(mdl3b.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3b failed!")        
if not np.array_equal(np.round(np.array(mdl3b.getValueVector()),6),np.round(np.array([200.0013199959708, 212.86689665815788, 298.709197760198]),6)):
        sys.exit("Model 3b failed!")

#Model 1c
mdl1c = mdpsolver.model()
mdl1c.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl1c.solve(algorithm="mpi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1c failed!")        
if not np.array_equal(np.round(np.array(mdl1c.getValueVector()),6),np.round(np.array([200.0012771893156, 212.86686671568802, 298.7091798650016]),6)):
        sys.exit("Model 1c failed!")

#Model 2c
mdl2c = mdpsolver.model()
mdl2c.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl2c.solve(algorithm="pi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2c failed!")        
if not np.array_equal(np.round(np.array(mdl2c.getValueVector()),6),np.round(np.array([200.0012771893156, 212.86686671568802, 298.7091798650016]),6)):
        sys.exit("Model 2c failed!")

#Model 3c
mdl3c = mdpsolver.model()
mdl3c.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl3c.solve(algorithm="vi",
          update="gs",
          parallel=False)
if not np.array_equal(np.array(mdl3c.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3c failed!")        
if not np.array_equal(np.round(np.array(mdl3c.getValueVector()),6),np.round(np.array([200.0010604606014, 212.86664512773723, 298.7089561808565]),6)):
        sys.exit("Model 3c failed!")

#Model 1d
mdl1d = mdpsolver.model()
mdl1d.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl1d.solve(algorithm="mpi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl1d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 1d failed!")        
if not np.array_equal(np.round(np.array(mdl1d.getValueVector()),6),np.round(np.array([200.00127021272166, 212.86686059325598, 298.70917425463125]),6)):
        sys.exit("Model 1d failed!")

#Model 2d
mdl2d = mdpsolver.model()
mdl2d.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl2d.solve(algorithm="pi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
if not np.array_equal(np.array(mdl2d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 2d failed!")        
if not np.array_equal(np.round(np.array(mdl2d.getValueVector()),6),np.round(np.array([200.00127021272166, 212.86686059325598, 298.70917425463125]),6)):
        sys.exit("Model 2d failed!")

#Model 3d
mdl3d = mdpsolver.model()
mdl3d.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl3d.solve(algorithm="vi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False)
if not np.array_equal(np.array(mdl3d.getPolicy()),np.array([1,1,0])):
        sys.exit("Model 3d failed!")        
if not np.array_equal(np.round(np.array(mdl3d.getValueVector()),6),np.round(np.array([206.69740452375487, 222.30968454932525, 305.91165352015094]),6)):
        sys.exit("Model 3d failed!")

print("Test 1 succesfully reproduced output!")