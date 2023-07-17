import mdpsolver

#mat = [[0.05,0.40,0.30,0.20,0.05,0.00,0.00,0.00,0.00,0.00,0.00],
#       [0.05,0.40,0.30,0.20,0.05,0.00,0.00,0.00,0.00,0.00,0.00]]

mdpsolver.tbm(discount=0.99,components=2,stages=10)
#mdpsolver.cbm(discount=0.99,components=2,stages=10,pCompMat=mat)

mdpsolver.solve()

#mdpsolver.printPolicy()