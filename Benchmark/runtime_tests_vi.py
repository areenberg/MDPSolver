import mdpsolver
import mdptoolbox
import time
import random
import math
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix


# --------------------------------------------
#   FUNCTIONS
# --------------------------------------------


def sampleProbs(n):
    rnds = [random.expovariate(1) for _ in range(n)]
    sm = sum(rnds)
    rnds_norm = [x / sm for x in rnds]
    return rnds_norm


def randomMDP(nStates, nActions, sparsity):

    nJumps = math.ceil(nStates * (1 - sparsity))

    # mdpsolver format
    tranMatProbs_mdpsolver = [[[] for _ in range(nActions)] for _ in range(nStates)]
    tranMatColumns_mdpsolver = [[[] for _ in range(nActions)] for _ in range(nStates)]
    rewards_mdpsolver = [[0 for _ in range(nActions)] for _ in range(nStates)]

    rndrew = 0
    for sidx in range(nStates):
        for aidx in range(nActions):
            rndrew = random.gauss(rndrew, 1)
            rewards_mdpsolver[sidx][aidx] = rndrew

            probs = sampleProbs(nJumps)
            cols = random.sample(range(nStates), nJumps)
            tranMatProbs_mdpsolver[sidx][aidx] = probs
            tranMatColumns_mdpsolver[sidx][aidx] = sorted(cols)

    # mdptoolbox format
    transitions_toolbox = np.empty(nActions, dtype=object)
    rewards_toolbox = np.zeros((nStates, nActions))
    for aidx in range(nActions):
        vals = np.zeros(nStates * nJumps)
        rowIndices = np.zeros(nStates * nJumps)
        colIndices = np.zeros(nStates * nJumps)
        k = 0
        for sidx in range(nStates):
            rewards_toolbox[sidx][aidx] = rewards_mdpsolver[sidx][aidx]
            for jidx in range(nJumps):
                vals[k] = tranMatProbs_mdpsolver[sidx][aidx][jidx]
                rowIndices[k] = sidx
                colIndices[k] = tranMatColumns_mdpsolver[sidx][aidx][jidx]
                k += 1
        mat_coo = coo_matrix((vals, (rowIndices, colIndices)), shape=(nStates, nStates))
        mat_csr = mat_coo.tocsr()
        transitions_toolbox[aidx] = mat_csr

    return (
        rewards_mdpsolver,
        tranMatProbs_mdpsolver,
        tranMatColumns_mdpsolver,
        rewards_toolbox,
        transitions_toolbox,
    )


# --------------------------------------------
#   TESTS
# --------------------------------------------

# discount
discount = 0.99
# tolerance
tolerance = 1e-3
# parIter (MPI only)
parIterLim = 100
# state space sizes
nStates = [2500, 5000, 10000]
# actions
nActions_min = 2
nActions_max = 10
# sparsity
sparsity = [[0.8, 0.9, 0.99], [0.8, 0.9, 0.99], [0.8, 0.9, 0.99]]
# replications
reps = 20
# seed
random.seed(123)

results = []
iter = 0

nsidx = 0
for ns in nStates:
    for sp in sparsity[nsidx]:
        r = 0
        while r < reps:
            nActions = random.randint(nActions_min, nActions_max)
            (
                rewards_mdpsolver,
                tranMatProbs_mdpsolver,
                tranMatColumns_mdpsolver,
                rewards_toolbox,
                transitions_toolbox,
            ) = randomMDP(ns, nActions, sp)

            # mdpsolver
            mdl = mdpsolver.model()
            mdl.mdp(
                discount=discount,
                rewards=rewards_mdpsolver,
                tranMatProbs=tranMatProbs_mdpsolver,
                tranMatColumns=tranMatColumns_mdpsolver,
            )
            initVal = [0] * ns
            initPol = [0] * ns
            mdl.solve(
                algorithm="vi",
                update="standard",
                tolerance=tolerance,
                initValueVector=initVal,
                initPolicy=initPol,
                verbose=False,
                postProcessing=False,
                makeFinalCheck=False,
            )
            runtime_mdpsolver = mdl.getRuntime()
            # print("mdpsolver:",runtime_mdpsolver)

            # mdptoolbox
            mdl_toolbox = mdptoolbox.mdp.ValueIteration(
                transitions_toolbox, rewards_toolbox, discount, epsilon=tolerance
            )
            start_time = time.time()
            mdl_toolbox.run()
            end_time = time.time()
            runtime_toolbox = (end_time - start_time) * 1000
            # print("toolbox:",runtime_toolbox)

            reldiff = runtime_toolbox / runtime_mdpsolver
            if reldiff > 0.0:
                # store results
                results.append(
                    [
                        sp,
                        ns,
                        nActions,
                        r,
                        runtime_mdpsolver,
                        runtime_toolbox,
                        reldiff,
                        "VI",
                    ]
                )
                iter += 1
                r += 1
                print(
                    str(iter) + " / " + str(len(nStates) * len(sparsity) * 3 * reps),
                    "ns:",
                    ns,
                    "sp:",
                    sp,
                    "rd:",
                    reldiff,
                )
            else:
                print("Failed.")
    nsidx += 1

# save as csv-file
df = pd.DataFrame(
    results,
    columns=[
        "Sparsity",
        "NumStates",
        "NumActions",
        "Replication",
        "RuntimeMDPSolver",
        "RuntimeToolbox",
        "RelativeDifference",
        "Algorithm",
    ],
)
df.to_csv("results/mdp_test_results_vi.csv", index=False)
