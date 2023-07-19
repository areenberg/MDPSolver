import mdpsolver



#mdl1 = mdpsolver.Model()
#mdl1.tbm(discount=0.99,components=2,stages=10)
#mdl1.solve()


mdl2 = mdpsolver.Model()
mat = [[0.05,0.40,0.30,0.20,0.05,0.00,0.00,0.00,0.00,0.00,0.00],
       [0.05,0.40,0.30,0.20,0.05,0.00,0.00,0.00,0.00,0.00,0.00]]
mdl2.cbm(discount=0.99,components=2,stages=10,pCompMat=mat)
mdl2.solve()

mdl2.printPolicy()