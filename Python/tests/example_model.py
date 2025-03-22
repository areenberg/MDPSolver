import numpy as np
import os

# Create transition probability matrix and rewards for the 'mdpsolver' examples.


# -----------------------------
#   FUNCTIONS
# -----------------------------


def exitDist(startK, arrRate, serRate, tol=1e-9):
    # calculate distribution over states observed by
    # the next arrival assuming the system currently
    # contains 'startK' customers.

    g = [i * serRate / (i * serRate + arrRate) for i in range(startK, -1, -1)]
    d = [arrRate / (i * serRate + arrRate) for i in range(startK, -1, -1)]

    Dmat = np.diag(d)
    Zmat = np.zeros((len(g), len(d)))
    Imat = np.eye(len(g))
    Gmat = np.zeros((len(g), len(g)))
    for i in range(startK):
        Gmat[i, i + 1] = g[i]
    mat = np.block([[Gmat, Dmat], [Zmat, Imat]])
    p = np.zeros((1, (startK + 1) * 2))
    p[0, 0] = 1

    while np.sum(p[0, : startK + 1]) >= tol:
        p = np.matmul(p, mat)

    return p[0, startK + 1 :]


def tranMat(arrRate1, arrRate2, serRate, capacity, fileName):
    # calculate transition probabilities and
    # and save them in a file

    ns = capacity + 1
    arrRate = arrRate1 + arrRate2
    pc1 = arrRate1 / arrRate
    pc2 = 1 - pc1

    if os.path.exists(fileName):
        os.remove(fileName)
    with open(fileName, "a") as file:
        file.write(f"stateFrom,action,stateTo,probability\n")
        for sidx in range(ns):  # stateFrom (customer type 1)
            for aidx in range(2):
                if aidx == 0:  # reject
                    if sidx > 0:
                        dist = exitDist(sidx, arrRate, serRate)
                    else:
                        dist = np.array([1])
                    dist1 = pc1 * dist
                    dist2 = pc2 * dist
                    for jidx in range(sidx + 1):
                        out = [sidx, aidx, jidx, dist1[sidx - jidx]]
                        str_out = ",".join(str(num) for num in out)
                        file.write(str_out + "\n")
                    for jidx in range(ns, ns + sidx + 1):
                        out = [sidx, aidx, jidx, dist2[sidx - (jidx - ns)]]
                        str_out = ",".join(str(num) for num in out)
                        file.write(str_out + "\n")
                elif aidx == 1 and sidx < capacity:  # accept
                    startK = sidx + 1
                    dist = exitDist(startK, arrRate, serRate)
                    dist1 = pc1 * dist
                    dist2 = pc2 * dist
                    for jidx in range(startK + 1):
                        out = [sidx, aidx, jidx, dist1[startK - jidx]]
                        str_out = ",".join(str(num) for num in out)
                        file.write(str_out + "\n")
                    for jidx in range(ns, ns + startK + 1):
                        out = [sidx, aidx, jidx, dist2[startK - (jidx - ns)]]
                        str_out = ",".join(str(num) for num in out)
                        file.write(str_out + "\n")

        for sidx in range(ns):  # stateFrom (customer type 2)
            for aidx in range(2):
                if aidx == 0:  # reject
                    if sidx > 0:
                        dist = exitDist(sidx, arrRate, serRate)
                    else:
                        dist = np.array([1])
                    dist1 = pc1 * dist
                    dist2 = pc2 * dist
                    for jidx in range(sidx + 1):
                        out = [sidx + ns, aidx, jidx, dist1[sidx - jidx]]
                        str_out = ",".join(str(num) for num in out)
                        file.write(str_out + "\n")
                    for jidx in range(ns, ns + sidx + 1):
                        out = [sidx + ns, aidx, jidx, dist2[sidx - (jidx - ns)]]
                        str_out = ",".join(str(num) for num in out)
                        file.write(str_out + "\n")
                elif aidx == 1 and sidx < capacity:  # accept
                    startK = sidx + 1
                    dist = exitDist(startK, arrRate, serRate)
                    dist1 = pc1 * dist
                    dist2 = pc2 * dist
                    for jidx in range(startK + 1):
                        out = [sidx + ns, aidx, jidx, dist1[startK - jidx]]
                        str_out = ",".join(str(num) for num in out)
                        file.write(str_out + "\n")
                    for jidx in range(ns, ns + startK + 1):
                        out = [sidx + ns, aidx, jidx, dist2[startK - (jidx - ns)]]
                        str_out = ",".join(str(num) for num in out)
                        file.write(str_out + "\n")


def rewards(capacity, rewardCust1, rewardCust2, fileName):

    ns = capacity + 1

    if os.path.exists(fileName):
        os.remove(fileName)
    with open(fileName, "a") as file:
        file.write(f"stateFrom,action,reward\n")
        for sidx in range(ns):  # stateFrom (customer type 1)
            for aidx in range(2):
                if aidx == 0:
                    out = [sidx, aidx, 0]
                    str_out = ",".join(str(num) for num in out)
                    file.write(str_out + "\n")
                elif aidx == 1 and sidx < capacity:
                    out = [sidx, aidx, rewardCust1]
                    str_out = ",".join(str(num) for num in out)
                    file.write(str_out + "\n")
        for sidx in range(ns):  # stateFrom (customer type 2)
            for aidx in range(2):
                if aidx == 0:
                    out = [sidx + ns, aidx, 0]
                    str_out = ",".join(str(num) for num in out)
                    file.write(str_out + "\n")
                elif aidx == 1 and sidx < capacity:
                    out = [sidx + ns, aidx, rewardCust2]
                    str_out = ",".join(str(num) for num in out)
                    file.write(str_out + "\n")
