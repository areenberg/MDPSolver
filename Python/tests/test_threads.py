'''
UNIT TEST FOR SOLVING IN PYTHON THREADS.

solve() releases Python's global interpreter lock (GIL) while it computes, since
the computation touches no Python objects. So other Python code runs while a
model is solved, and model objects solved in several Python threads are solved
at the same time. (Until 0.11.0, solve() held the GIL throughout, which stopped
every other Python thread until it was done.) The calls on one model object are
run one at a time, by a lock in each model object, so that no call reads or
changes a model while another thread solves it.

The models are random, with a fixed seed: every state has the same number of
actions and every state-action pair the same number of next states, given to
mdp() as NumPy arrays.

Sub-tests:
    Case 1: Python code runs while solve() computes: a loop in the main thread,
            run while another thread solves a model, makes at least a tenth of
            the progress it makes alone in the same time.
    Case 2: Models solved in several threads at once give exactly the
            policies, values and numbers of iterations they give solved one
            after another, under each algorithm, with the parallel computation
            on and off.
    Case 3: One model object used by several threads at once, two of them
            solving it with different settings and two reading its results
            meanwhile, does not crash, and every policy and value vector read
            is exactly that of one of the settings solved alone: never one of a
            run still in progress.

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from loadmdpsolver import mdpsolver, finalline


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

DISCOUNT = 0.99
TOLERANCE = 1e-6             # the epsilon handed to solve()
MINPROGRESS = 0.1            # case 1: the share of its own pace the main thread must keep
MINSOLVETIME = 50.0          # case 1: the solve must take at least this many milliseconds to tell anything
NTHREADS = 4
ALGORITHMS = ("vi", "pi", "mpi")

# Case 1: a model whose solve takes a while (value iteration with Gauss-Seidel
# updates, serial); cases 2 and 3: smaller ones
LONGMODEL = (5000, 5, 20)
SMALLMODEL = (2000, 4, 25)

# Case 3: the two settings the solving threads take turns with; the initial
# policy and values are given, so that each run gives the same results
# whichever runs came before it
SETTINGS = (dict(algorithm="vi", update="gs", tolerance=1e-6, parallel=False),
            dict(algorithm="mpi", update="standard", tolerance=1e-3, parallel=True))
NROUNDS = 4

warnings.simplefilter("error")   # a run that does not converge would make the comparisons void


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def randommodel(S, A, K, seed):
    '''The arguments of mdp() for a random MDP: K next states per state-action pair, all reachable.'''
    rng = np.random.default_rng(seed)
    probs = rng.exponential(size=(S, A, K))
    probs /= probs.sum(axis=2, keepdims=True)
    columns = rng.integers(0, S, size=(S, A, K))
    return dict(rewards=rng.normal(size=(S, A)), tranMatProbs=probs, tranMatColumns=columns)


def newmodel(arguments):
    mdl = mdpsolver.model()
    mdl.mdp(discount=DISCOUNT, **arguments)
    return mdl


def solved(arguments, **setting):
    '''A new model object with the model solved; returns the policy, the values and the number of iterations.'''
    mdl = newmodel(arguments)
    mdl.solve(**{"tolerance": TOLERANCE, **setting})
    return mdl.getPolicy().tolist(), mdl.getValueVector().tolist(), mdl.getIterations()


def pace(until):
    '''Counts the turns of a loop in this thread until the event is set; returns the count and the seconds taken.'''
    count = 0
    start = time.perf_counter()
    while not until.is_set():
        count += 1
    return count, time.perf_counter() - start


# ------------------------------------------------------------------
# CASE 1: Python code runs while solve() computes
# ------------------------------------------------------------------

LONG = newmodel(randommodel(*LONGMODEL, seed=1))

# The main thread's own pace: the loop runs until a timer stops it
alone = threading.Event()
timer = threading.Timer(0.2, alone.set)
timer.start()
alonecount, alonetime = pace(alone)
timer.join()

# Its progress while another thread solves the model, over the time from
# just before that thread starts to the end of its solve. (Holding the GIL, the
# solve blocks the main thread even in Thread.start(), so the main thread's own
# clock would miss the time it was blocked.)
done = threading.Event()
window = {}
def solvelong():
    try:
        LONG.solve(algorithm="vi", update="gs", tolerance=TOLERANCE, parallel=False)
    finally:
        window["end"] = time.perf_counter()
        done.set()
solver = threading.Thread(target=solvelong)
window["start"] = time.perf_counter()
solver.start()
duringcount, _ = pace(done)
solver.join()

if LONG.getRuntime() < MINSOLVETIME:
    sys.exit("Validation test failed at case 1: the solve took only %.1f ms, too little to tell whether other threads run meanwhile; enlarge LONGMODEL." % LONG.getRuntime())
progress = (duringcount / (window["end"] - window["start"])) / (alonecount / alonetime)
if progress < MINPROGRESS:
    sys.exit("Validation test failed at case 1: while another thread solved a model for %.0f ms, the main thread kept only %.1f%% of its own pace, so solve() holds the GIL." % (LONG.getRuntime(), 100 * progress))


# ------------------------------------------------------------------
# CASE 2: Models solved in several threads at once
# ------------------------------------------------------------------

MODELS = [randommodel(*SMALLMODEL, seed=seed) for seed in range(NTHREADS)]
for algorithm in ALGORITHMS:
    for parallel in (False, True):
        setting = dict(algorithm=algorithm, parallel=parallel)
        oneafteranother = [solved(arguments, **setting) for arguments in MODELS]
        with ThreadPoolExecutor(NTHREADS) as pool:
            atonce = list(pool.map(lambda arguments: solved(arguments, **setting), MODELS))
        for k, (first, second) in enumerate(zip(oneafteranother, atonce)):
            if first != second:
                what = [name for name, x, y in zip(("policy", "values", "iterations"), first, second) if x != y]
                sys.exit("Validation test failed at case 2: model %d under %s gives other %s when solved in threads at once than when solved one after another." % (k, setting, " and ".join(what)))


# ------------------------------------------------------------------
# CASE 3: One model object used by several threads at once
# ------------------------------------------------------------------

SHARED = MODELS[0]
NSMALL = SMALLMODEL[0]
INITIAL = dict(initPolicy=np.zeros(NSMALL, dtype=int), initValueVector=np.zeros(NSMALL))
EXPECTED = []
for setting in SETTINGS:
    mdl = newmodel(SHARED)
    mdl.solve(**INITIAL, **setting)
    EXPECTED.append((mdl.getPolicy().tolist(), mdl.getValueVector().tolist()))
if EXPECTED[0][1] == EXPECTED[1][1]:
    sys.exit("Validation test failed at initialization: the two settings of case 3 give the same values, so a mixed-up read could not be told.")

shared = newmodel(SHARED)
shared.solve(**INITIAL, **SETTINGS[0])
finished = threading.Event()
errors, reads = [], []

def solving(first):
    try:
        for round in range(NROUNDS):
            shared.solve(**INITIAL, **SETTINGS[(first + round) % 2])
    except Exception as error:
        errors.append(error)

def reading():
    try:
        while not finished.is_set():
            reads.append((shared.getPolicy().tolist(), shared.getValueVector().tolist()))
            shared.getAction(stateIndex=NSMALL - 1)
            shared.getValue(stateIndex=0)
    except Exception as error:
        errors.append(error)

solvers = [threading.Thread(target=solving, args=(first,)) for first in (0, 1)]
readers = [threading.Thread(target=reading) for _ in range(2)]
for thread in readers + solvers:
    thread.start()
for thread in solvers:
    thread.join()
finished.set()
for thread in readers:
    thread.join()

if errors:
    sys.exit("Validation test failed at case 3: a thread using the shared model object raised %r." % errors[0])
if not reads:
    sys.exit("Validation test failed at case 3: the reading threads read nothing.")
policies = [expected[0] for expected in EXPECTED]
values = [expected[1] for expected in EXPECTED]
for k, (policy, value) in enumerate(reads):
    if policy not in policies or value not in values:
        sys.exit("Validation test failed at case 3: read %d of %d gave a %s that no setting gives when solved alone, i.e. one of a run still in progress." % (k + 1, len(reads), "policy" if policy not in policies else "value vector"))


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
