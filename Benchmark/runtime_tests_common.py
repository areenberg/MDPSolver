"""
RUNTIME TESTS: MDPSOLVER AGAINST PYMDPTOOLBOX

Shared by runtime_tests_vi.py, runtime_tests_pi.py and runtime_tests_mpi.py, which
each run one algorithm. The results are described on the wiki page "Performance
experiments", and plot_results.py draws its figures.

The models are random MDPs with nStates states and the same number of actions in
every state (pymdptoolbox needs that), between 2 and 10, drawn for each model. Every
state-action pair has ceil(nStates * (1 - sparsity)) next states, drawn uniformly
without replacement, with exponentially distributed probabilities (normalised to sum
to 1). The rewards are a Gaussian random walk over the state-action pairs: each
reward is the previous one plus a standard normal number.

Each model is solved by MDPSolver twice, in parallel (its default) and on one thread,
and by pymdptoolbox. The runtimes are those of the algorithms alone: MDPSolver's
getRuntime(), which times solve(), and the wall time of pymdptoolbox's run(). Neither
includes loading the model (mdp(), and the constructor of pymdptoolbox's algorithm).
"""

import math
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import mdpsolver
import mdptoolbox
import mdptoolbox.mdp
import mdptoolbox.util

# pymdptoolbox checks the model in the constructor of every algorithm (that the
# transition matrices are square, stochastic and non-negative, and the rewards of the
# right shape), and its newest version, 4.0b3, has no option to skip the check.
# MDPSolver checks a model only when asked (checkModel()), so pymdptoolbox's check is
# switched off here. The check never counted in the runtimes, which time run() only;
# switching it off shortens the tests.
mdptoolbox.util.check = lambda transitions, reward: None


def boundIter(self, epsilon):
    """pymdptoolbox 4.0b3's ValueIteration._boundIter (the largest number of
    iterations of value iteration, set in its constructor), with the smallest entry
    of each column found per matrix rather than column by column, which takes minutes
    for the larger models here. Otherwise line for line the same, and so the same
    limit (checked on random models), and the same run()."""
    h = np.min([np.asarray(self.P[aa].min(axis=0).todense()).ravel() for aa in range(self.A)], axis=0)
    k = 1 - h.sum()
    Vprev = self.V
    null, value = self._bellmanOperator()
    span = mdptoolbox.util.getSpan(value - Vprev)
    max_iter = (math.log((epsilon * (1 - self.discount) / self.discount) /
                span) / math.log(self.discount * k))
    self.max_iter = int(math.ceil(max_iter))


originalBoundIter = mdptoolbox.mdp.ValueIteration._boundIter
mdptoolbox.mdp.ValueIteration._boundIter = boundIter

DISCOUNT = 0.99
STATES = (2500, 5000, 10000)
SPARSITIES = (0.8, 0.9, 0.99)
ACTIONS = (2, 10)  # the smallest and largest number of actions
REPLICATIONS = 20
SEED = 123
ROWS = 1000  # state-action pairs drawn at a time


def randomMDP(rng, nStates, nActions, sparsity):
    """A random model: rewards (states x actions), and the probabilities and next
    states (states x actions x next states per pair) as NumPy arrays."""
    nJumps = math.ceil(nStates * (1 - sparsity))
    pairs = nStates * nActions
    rewards = np.cumsum(rng.normal(size=pairs)).reshape(nStates, nActions)
    probs = rng.exponential(size=(pairs, nJumps))
    probs /= probs.sum(axis=1, keepdims=True)
    columns = np.empty((pairs, nJumps), dtype=np.int64)
    for start in range(0, pairs, ROWS):
        stop = min(pairs, start + ROWS)
        # nJumps distinct next states per pair: those with the smallest random keys
        keys = rng.random((stop - start, nStates))
        columns[start:stop] = np.sort(np.argpartition(keys, nJumps - 1, axis=1)[:, :nJumps], axis=1)
    return (rewards, probs.reshape(nStates, nActions, nJumps),
            columns.reshape(nStates, nActions, nJumps))


def toolboxModel(probs, columns):
    """The transition matrices in pymdptoolbox's format: one sparse (CSR) matrix per
    action, in an array of objects."""
    nStates, nActions, nJumps = probs.shape
    rowStart = np.arange(0, nStates * nJumps + 1, nJumps)
    transitions = np.empty(nActions, dtype=object)
    for aidx in range(nActions):
        transitions[aidx] = csr_matrix(
            (probs[:, aidx, :].reshape(-1), columns[:, aidx, :].reshape(-1).astype(np.int32), rowStart),
            shape=(nStates, nStates))
    return transitions


def runTests(name, mdpsolverSettings, toolboxAlgorithm):
    """Runs one algorithm on REPLICATIONS models of each number of states and
    sparsity, and saves the results to results/mdp_test_results_<name>.csv.

    mdpsolverSettings: the arguments of MDPSolver's solve() for the algorithm.
    toolboxAlgorithm(transitions, rewards): the algorithm of pymdptoolbox, created.
    """
    rng = np.random.default_rng(SEED)
    results = []
    total = len(STATES) * len(SPARSITIES) * REPLICATIONS
    for nStates in STATES:
        for sparsity in SPARSITIES:
            for r in range(REPLICATIONS):
                nActions = int(rng.integers(ACTIONS[0], ACTIONS[1] + 1))
                rewards, probs, columns = randomMDP(rng, nStates, nActions, sparsity)

                # MDPSolver, in parallel (the default) and on one thread
                runtime = {}
                for parallel in (True, False):
                    mdl = mdpsolver.model()
                    mdl.mdp(discount=DISCOUNT, rewards=rewards, tranMatProbs=probs, tranMatColumns=columns)
                    mdl.solve(**mdpsolverSettings,
                              initValueVector=np.zeros(nStates),
                              initPolicy=np.zeros(nStates, dtype=np.int64),
                              verbose=False,
                              postProcessing=False,
                              makeFinalCheck=False,
                              parallel=parallel)
                    runtime[parallel] = mdl.getRuntime()
                    iterations = mdl.getIterations()
                    policy = mdl.getPolicy()
                    del mdl

                # pymdptoolbox
                toolbox = toolboxAlgorithm(toolboxModel(probs, columns), rewards)
                start = time.perf_counter()
                toolbox.run()
                runtimeToolbox = (time.perf_counter() - start) * 1000
                agreement = float(np.mean(policy == np.asarray(toolbox.policy)))

                results.append([sparsity, nStates, nActions, r,
                                runtime[True], runtime[False], runtimeToolbox,
                                runtimeToolbox / runtime[True], runtimeToolbox / runtime[False],
                                iterations, toolbox.iter, agreement, name.upper()])
                print("%d / %d  states: %d  sparsity: %g  actions: %d  speedup: %.1f (one thread: %.1f)  same policy: %.4f" % (
                    len(results), total, nStates, sparsity, nActions,
                    results[-1][7], results[-1][8], agreement), flush=True)
                del toolbox, rewards, probs, columns

    df = pd.DataFrame(results, columns=[
        "Sparsity", "NumStates", "NumActions", "Replication",
        "RuntimeMDPSolver", "RuntimeMDPSolverOneThread", "RuntimeToolbox",
        "RelativeDifference", "RelativeDifferenceOneThread",
        "IterationsMDPSolver", "IterationsToolbox", "PolicyAgreement", "Algorithm"])
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, "mdp_test_results_%s.csv" % name), index=False)
    return df
