import mdpsolver

mdl = mdpsolver.Model()

rewards = [[0.5,-0.1],
           [1.0,-0.2],
           [0.0,50.0]]
tranMatWithZeros = [[[0.9,0.1,0.0],[0.1,0.9,0.0]],
                    [[0.4,0.5,0.1],[0.3,0.5,0.2]],
                    [[0.9,0.1,0.0],[0.2,0.6,0.2]]]

mdl.mdp(discount=0.8,
        rewards=rewards,
        tranMatWithZeros=tranMatWithZeros)

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