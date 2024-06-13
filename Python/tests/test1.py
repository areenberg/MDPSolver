import mdpsolver
import random
from random import randint

#TEST 1
#Simple MDP with 3 states and 2 actions in each state.

#---------------------------------------
# CONFIGURATION 1
#---------------------------------------

print("## CONFIG. 1 ##\n")

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
print("# Model 1a (discounted reward, parallel) #")
mdl1a = mdpsolver.model()
mdl1a.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1a.solve(algorithm="mpi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
print(mdl1a.getPolicy())
print(mdl1a.getValueVector())

#Model 2a (discounted reward, parallel)
print("# Model 2a (discounted reward, parallel) #")
mdl2a = mdpsolver.model()
mdl2a.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2a.solve(algorithm="pi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
print(mdl2a.getPolicy())
print(mdl2a.getValueVector())

#Model 3a (discounted reward, parallel)
print("# Model 3a (discounted reward, parallel) #")
mdl3a = mdpsolver.model()
mdl3a.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl3a.solve(algorithm="vi",
          update="standard",
          parallel=True)
print(mdl3a.getPolicy())
print(mdl3a.getValueVector())

#Model 1b (discounted reward, unparallel)
print("# Model 1b (discounted reward, unparallel) #")
mdl1b = mdpsolver.model()
mdl1b.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1b.solve(algorithm="mpi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
print(mdl1b.getPolicy())
print(mdl1b.getValueVector())

#Model 2b (discounted reward, unparallel)
print("# Model 2b (discounted reward, unparallel) #")
mdl2b = mdpsolver.model()
mdl2b.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2b.solve(algorithm="pi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
print(mdl2b.getPolicy())
print(mdl2b.getValueVector())

#Model 3b (discounted reward, unparallel)
print("# Model 3b (discounted reward, unparallel) #")
mdl3b = mdpsolver.model()
mdl3b.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl3b.solve(algorithm="vi",
          update="standard",
          parallel=False)
print(mdl3b.getPolicy())
print(mdl3b.getValueVector())

#Model 1a (average reward, parallel)
print("# Model 1a (average reward, parallel) #")
mdl1a = mdpsolver.model()
mdl1a.mdp(rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1a.solve(algorithm="mpi",
          update="standard",
          criterion="average",
          parallel=True,
          initPolicy=initPolicy)
print(mdl1a.getPolicy())
print(mdl1a.getValueVector())

#Model 2a (average reward, parallel)
print("# Model 2a (average reward, parallel) #")
mdl2a = mdpsolver.model()
mdl2a.mdp(rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2a.solve(algorithm="pi",
          update="standard",
          criterion="average",
          parallel=True,
          initPolicy=initPolicy)
print(mdl2a.getPolicy())
print(mdl2a.getValueVector())

#Model 3a (average reward, parallel)
print("# Model 3a (average reward, parallel) #")
mdl3a = mdpsolver.model()
mdl3a.mdp(rewards=rewards,
          tranMatWithZeros=tranMatWithZeros)
mdl3a.solve(algorithm="vi",
          update="standard",
          criterion="average",
          parallel=True)
print(mdl3a.getPolicy())
print(mdl3a.getValueVector())

#Model 1b (average reward, unparallel)
print("# Model 1b (average reward, unparallel) #")
mdl1b = mdpsolver.model()
mdl1b.mdp(rewards=rewards,
          tranMatWithZeros=tranMatWithZeros)
mdl1b.solve(algorithm="mpi",
          update="standard",
          criterion="average",
          parallel=False,
          initPolicy=initPolicy)
print(mdl1b.getPolicy())
print(mdl1b.getValueVector())

#Model 2b (average reward, unparallel)
print("# Model 2b (average reward, unparallel) #")
mdl2b = mdpsolver.model()
mdl2b.mdp(rewards=rewards,
          tranMatWithZeros=tranMatWithZeros)
mdl2b.solve(algorithm="pi",
          update="standard",
          criterion="average",
          parallel=False,
          initPolicy=initPolicy)
print(mdl2b.getPolicy())
print(mdl2b.getValueVector())

#Model 3b (average reward, unparallel)
print("# Model 3b (average reward, unparallel) #")
mdl3b = mdpsolver.model()
mdl3b.mdp(rewards=rewards,
          tranMatWithZeros=tranMatWithZeros)
mdl3b.solve(algorithm="vi",
          update="standard",
          criterion="average",
          parallel=False)
print(mdl3b.getPolicy())
print(mdl3b.getValueVector())

#Model 1c (discounted, Gauss-Seidel)
print("# Model 1c (discounted reward, Gauss-Seidel) #")
mdl1c = mdpsolver.model()
mdl1c.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1c.solve(algorithm="mpi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
print(mdl1c.getPolicy())
print(mdl1c.getValueVector())

#Model 2c (Gauss-Seidel)
print("# Model 2c (discounted reward, Gauss-Seidel) #")
initPolicy=[1,1,1]
mdl2c = mdpsolver.model()
mdl2c.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2c.solve(algorithm="pi",
            update="gs",
            parallel=False,
            initPolicy=initPolicy)
print(mdl2c.getPolicy())
print(mdl2c.getValueVector())

#Model 3c (Gauss-Seidel)
print("# Model 3c (discounted reward, Gauss-Seidel) #")
mdl3c = mdpsolver.model()
mdl3c.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl3c.solve(algorithm="vi",
          update="gs",
          parallel=False)
print(mdl3c.getPolicy())
print(mdl3c.getValueVector())

#Model 1d (SOR)
print("# Model 1d (discounted reward, SOR) #")
mdl1d = mdpsolver.model()
mdl1d.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl1d.solve(algorithm="mpi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
print(mdl1d.getPolicy())
print(mdl1d.getValueVector())

#Model 2d (SOR)
print("# Model 2d (discounted reward, SOR) #")
mdl2d = mdpsolver.model()
mdl2d.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl2d.solve(algorithm="pi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
print(mdl2d.getPolicy())
print(mdl2d.getValueVector())

#Model 3d (SOR)
print("# Model 3d (discounted reward, SOR) #")
mdl3d = mdpsolver.model()
mdl3d.mdp(discount=0.95,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)
mdl3d.solve(algorithm="vi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False)
print(mdl3d.getPolicy())
print(mdl3d.getValueVector())

#---------------------------------------
# CONFIGURATION 2
#---------------------------------------

print("\n ## CONFIG. 2 ##\n")

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
print("# Model 1a #")
mdl1a = mdpsolver.model()
mdl1a.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl1a.solve(algorithm="mpi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
print(mdl1a.getPolicy())
print(mdl1a.getValueVector())

#Model 2a
print("# Model 2a #")
mdl2a = mdpsolver.model()
mdl2a.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl2a.solve(algorithm="pi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
print(mdl2a.getPolicy())
print(mdl2a.getValueVector())

#Model 3a
print("# Model 3a #")
mdl3a = mdpsolver.model()
mdl3a.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl3a.solve(algorithm="vi",
          update="standard",
          parallel=True)
print(mdl3a.getPolicy())
print(mdl3a.getValueVector())

#Model 1b
print("# Model 1b #")
mdl1b = mdpsolver.model()
mdl1b.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl1b.solve(algorithm="mpi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
print(mdl1b.getPolicy())
print(mdl1b.getValueVector())

#Model 2b
print("# Model 2b #")
mdl2b = mdpsolver.model()
mdl2b.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl2b.solve(algorithm="pi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
print(mdl2b.getPolicy())
print(mdl2b.getValueVector())

#Model 3b
print("# Model 3b #")
mdl3b = mdpsolver.model()
mdl3b.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl3b.solve(algorithm="vi",
          update="standard",
          parallel=False)
print(mdl3b.getPolicy())
print(mdl3b.getValueVector())

#Model 1c
print("# Model 1c #")
mdl1c = mdpsolver.model()
mdl1c.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl1c.solve(algorithm="mpi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
print(mdl1c.getPolicy())
print(mdl1c.getValueVector())

#Model 2c
print("# Model 2c #")
mdl2c = mdpsolver.model()
mdl2c.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl2c.solve(algorithm="pi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
print(mdl2c.getPolicy())
print(mdl2c.getValueVector())

#Model 3c
print("# Model 3c #")
mdl3c = mdpsolver.model()
mdl3c.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl3c.solve(algorithm="vi",
          update="gs",
          parallel=False)
print(mdl3c.getPolicy())
print(mdl3c.getValueVector())

#Model 1d
print("# Model 1d #")
mdl1d = mdpsolver.model()
mdl1d.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl1d.solve(algorithm="mpi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
print(mdl1d.getPolicy())
print(mdl1d.getValueVector())

#Model 2d
print("# Model 2d #")
mdl2d = mdpsolver.model()
mdl2d.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl2d.solve(algorithm="pi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False,
          initPolicy=initPolicy)
print(mdl2d.getPolicy())
print(mdl2d.getValueVector())

#Model 3d
print("# Model 3d #")
mdl3d = mdpsolver.model()
mdl3d.mdp(discount=0.95,
        rewards=rewards,
        tranMatElementwise=tranMatElementwise)
mdl3d.solve(algorithm="vi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False)
print(mdl3d.getPolicy())
print(mdl3d.getValueVector())

#---------------------------------------
# CONFIGURATION 3
#---------------------------------------

print("\n ## CONFIG. 3 ##\n")

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
print("# Model 1a #")
mdl1a = mdpsolver.model()
mdl1a.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl1a.solve(algorithm="mpi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
print(mdl1a.getPolicy())
print(mdl1a.getValueVector())

#Model 2a
print("# Model 2a #")
mdl2a = mdpsolver.model()
mdl2a.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl2a.solve(algorithm="pi",
          update="standard",
          parallel=True,
          initPolicy=initPolicy)
print(mdl2a.getPolicy())
print(mdl2a.getValueVector())

#Model 3a
print("# Model 3a #")
mdl3a = mdpsolver.model()
mdl3a.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl3a.solve(algorithm="vi",
          update="standard",
          parallel=True)
print(mdl3a.getPolicy())
print(mdl3a.getValueVector())

#Model 1b
print("# Model 1b #")
mdl1b = mdpsolver.model()
mdl1b.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl1b.solve(algorithm="mpi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
print(mdl1b.getPolicy())
print(mdl1b.getValueVector())

#Model 2b
print("# Model 2b #")
mdl2b = mdpsolver.model()
mdl2b.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl2b.solve(algorithm="pi",
          update="standard",
          parallel=False,
          initPolicy=initPolicy)
print(mdl2b.getPolicy())
print(mdl2b.getValueVector())

#Model 3b
print("# Model 3b #")
mdl3b = mdpsolver.model()
mdl3b.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl3b.solve(algorithm="vi",
          update="standard",
          parallel=False)
print(mdl3b.getPolicy())
print(mdl3b.getValueVector())

#Model 1c
print("# Model 1c #")
mdl1c = mdpsolver.model()
mdl1c.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl1c.solve(algorithm="mpi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
print(mdl1c.getPolicy())
print(mdl1c.getValueVector())

#Model 2c
print("# Model 2c #")
mdl2c = mdpsolver.model()
mdl2c.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl2c.solve(algorithm="pi",
          update="gs",
          parallel=False,
          initPolicy=initPolicy)
print(mdl2c.getPolicy())
print(mdl2c.getValueVector())

#Model 3c
print("# Model 3c #")
mdl3c = mdpsolver.model()
mdl3c.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl3c.solve(algorithm="vi",
          update="gs",
          parallel=False)
print(mdl3c.getPolicy())
print(mdl3c.getValueVector())

#Model 1d
print("# Model 1d #")
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
print(mdl1d.getPolicy())
print(mdl1d.getValueVector())

#Model 2d
print("# Model 2d #")
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
print(mdl2d.getPolicy())
print(mdl2d.getValueVector())

#Model 3d
print("# Model 3d #")
mdl3d = mdpsolver.model()
mdl3d.mdp(discount=0.95,
        rewards=rewards,
        tranMatProbs=tranMatProbs,
        tranMatColumns=tranMatColumns)
mdl3d.solve(algorithm="vi",
          update="sor",
          SORrelaxation=1.01,
          parallel=False)
print(mdl3d.getPolicy())
print(mdl3d.getValueVector())