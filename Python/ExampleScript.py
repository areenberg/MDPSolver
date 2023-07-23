import mdpsolver

mdl = mdpsolver.Model()

rewards = [[0.5,-0.1],
           [1.0,-0.2],
           [0.0,50.0]]

#alternative representation
#rewardsElementwise = [[0,0,0.5],
#                      [0,1,-0.1],
#                      [1,0,1.0],
#                      [1,1,-0.2],
#                      [2,0,0.0],
#                      [2,1,50.0]]

tranMatWithZeros = [[[0.9,0.1,0.0],[0.1,0.9,0.0]],
                    [[0.4,0.5,0.1],[0.3,0.5,0.2]],
                    [[0.9,0.1,0.0],[0.2,0.6,0.2]]]

#alternative representation
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
#                      [2,0,0,0.9],
#                      [2,0,1,0.1],
#                      [2,1,0,0.2],
#                      [2,1,1,0.6],
#                      [2,1,2,0.2]]

mdl.mdp(discount=0.8,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)

#alternative representation
#mdl.mdp(discount=0.8,
#        rewardsElementwise=rewardsElementwise,
#        tranMatElementwise=tranMatElementwise)

mdl.solve(algorithm="pi",update="gs")
mdl.printPolicy()

#mdl1 = mdpsolver.Model()
#mdl1.tbm(discount=0.99,components=2,stages=10)
#mdl1.solve()

#mdl2 = mdpsolver.Model()
#mat = [[0.05,0.40,0.30,0.20,0.05,0.00,0.00,0.00,0.00,0.00,0.00],
#       [0.05,0.40,0.30,0.20,0.05,0.00,0.00,0.00,0.00,0.00,0.00]]

#mdl2.cbm(discount=0.99,components=2,stages=10,pCompMat=mat)
#mdl2.solve()

#mdl2.printPolicy()