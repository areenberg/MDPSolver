'''
UNIT TEST FOR NUMPY ARRAYS AS THE INPUT OF model.mdp() (and of the initial
vectors of model.solve()).

Every format of mdp() but the files can be given as a NumPy array as well as a
list:

    rewards              states x actions
    rewardsElementwise   n x 3: state, action, reward
    tranMatWithZeros     states x actions x states
    tranMatElementwise   n x 4: state, action, next state, probability
    tranMatProbs and     states x actions x K: the probabilities and the next
    tranMatColumns       states of each state-action pair, K of each

An array takes its own way into the internal representation, and must arrive
where the list of the same format arrives: at the same probabilities and next
states, in the same order. So the test requires the policies, the values and
the numbers of iterations from an array to equal those from the list exactly,
not just within the tolerance.

The MDP is random, with a fixed seed: every state has the same number of
actions, and every state-action pair the same number of next states, as the
array formats need, in random order. Every pair can lead to state 0, which can
stay in state 0, so that every policy is unichain and aperiodic (which value
iteration under the average-reward criterion needs), and some pairs can stay
in their own state (which the Gauss-Seidel and SOR updates handle apart from
the others).

Sub-tests:
    Case 1: The MDP given as arrays is solved to its exact solution, computed
            by policy iteration with numpy, under the three algorithms.
    Case 2: Each combination of a reward format and a transition format,
            given as arrays, gives exactly the results of the same formats
            given as lists, under the three algorithms with the three update
            methods (discounted) and with standard updates (average reward),
            in serial and in parallel.
    Case 3: The arrays need not be C-contiguous float64 (or int64 for the
            column indices): integer rewards, float32 probabilities, int32 or
            float column indices with whole values, Fortran order, views with
            strides, rows of tranMatProbs padded with probability 0 (with the
            state itself as the padding next state, the worst case for the
            Gauss-Seidel and SOR updates), and an array mixed with a list,
            also tranMatProbs with tranMatColumns, all give exactly the results
            of the lists, under the discounted settings of case 2.
    Case 4: An array that does not fit its format raises an error naming the
            argument and prints nothing besides: a wrong number of dimensions,
            tranMatProbs and tranMatColumns of different shapes, a list that
            cannot be made an array to go with one, a column index that is not
            a whole number, and an elementwise array with too few columns or a
            negative or fractional index raise a ValueError; an array of text
            a TypeError. The errors raised before anything is loaded leave the
            model object free for a valid mdp(). An empty array counts as not
            given, as an empty list does.
    Case 5: initPolicy and initValueVector given as arrays give exactly the
            results of the same vectors given as lists.
    Case 6: Transition probabilities given as float32 arrays, which the
            compiled module keeps in single precision, give exactly the results
            of the same numbers given as float64 arrays, in each of the three
            formats that hold probabilities, under the discounted settings of
            case 2. (Rounded to float32, the probabilities of a state-action
            pair sum to 1 only to within about 1e-7. Under the average reward
            criterion, whose iterates grow with every iteration, that keeps the
            solver from reaching the tolerance of 1e-8, with the numbers in
            float64 as well as in float32.)
            checkModel() finds no problem in such a model although its sums
            carry the rounding of float32, while it finds the sums of the same
            numbers given in float64 (so the float32 numbers are kept as such),
            and it still finds a sum of 0.9.

(getPolicy() and getValueVector() return NumPy arrays, which test_output.py
checks.)

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import os
import sys
import tempfile
import warnings

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import checkoptimality, exactdiscounted


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

NSTATES = 30
NACTIONS = 3
NNEXT = 6                    # next states per state-action pair
SEED = 11
DISCOUNT = 0.9
TOLERANCE = 1e-8             # the epsilon handed to solve()
MINGAP = 1e-3                # the best action of every state must win by this much in the exact solution
SORRELAXATION = 1.2

ALGORITHMS = ("vi", "pi", "mpi")
UPDATES = ("standard", "gs", "sor")
REWARDFORMATS = ("rewards", "rewardsElementwise")
TRANSITIONFORMATS = ("tranMatWithZeros", "tranMatElementwise", "tranMatProbs and tranMatColumns")

rng = np.random.default_rng(SEED)
REWARDS = rng.normal(size=(NSTATES, NACTIONS)) * 10.0
PROBS = rng.exponential(size=(NSTATES, NACTIONS, NNEXT))
PROBS /= PROBS.sum(axis=2, keepdims=True)
COLUMNS = np.zeros((NSTATES, NACTIONS, NNEXT), dtype=np.int64)
for s in range(NSTATES):
    for a in range(NACTIONS):
        # state 0, for every other pair the state itself, and other states, in random order
        stays = s > 0 and (s + a) % 2 == 0
        others = rng.choice([j for j in range(1, NSTATES) if j != s], size=NNEXT - 1 - stays, replace=False)
        COLUMNS[s, a] = rng.permutation(np.r_[0, [s] * stays, others])

# The transitions as dictionaries for knownsolutions.py, which also make sure
# that no next state appears twice in a pair
TRANSITIONS = [[{int(j): float(p) for j, p in zip(COLUMNS[s, a], PROBS[s, a])} for a in range(NACTIONS)] for s in range(NSTATES)]
if any(len(TRANSITIONS[s][a]) != NNEXT for s in range(NSTATES) for a in range(NACTIONS)):
    sys.exit("Validation test failed at initialization: a state-action pair has the same next state twice.")
if not all(0 in TRANSITIONS[s][a] for s in range(NSTATES) for a in range(NACTIONS)):
    sys.exit("Validation test failed at initialization: a state-action pair cannot lead to state 0.")
if not any(s in TRANSITIONS[s][a] for s in range(1, NSTATES) for a in range(NACTIONS)):
    sys.exit("Validation test failed at initialization: no pair can stay in its own state, so the Gauss-Seidel and SOR updates would not be put to the test.")

EXACTPOLICY, EXACTVALUES = exactdiscounted(REWARDS.tolist(), TRANSITIONS, DISCOUNT)
checkoptimality("random MDP of test_arrays", REWARDS.tolist(), TRANSITIONS, EXACTPOLICY, EXACTVALUES, DISCOUNT, MINGAP)

# Every format as arrays and as lists
DENSE = np.zeros((NSTATES, NACTIONS, NSTATES))
for s in range(NSTATES):
    for a in range(NACTIONS):
        DENSE[s, a, COLUMNS[s, a]] = PROBS[s, a]
ROWS = np.array([[s, a, j, p] for s in range(NSTATES) for a in range(NACTIONS) for j, p in zip(COLUMNS[s, a], PROBS[s, a])])
ROWS = ROWS[rng.permutation(len(ROWS))]
REWARDROWS = np.array([[s, a, REWARDS[s, a]] for s in range(NSTATES) for a in range(NACTIONS)])
REWARDROWS = REWARDROWS[rng.permutation(len(REWARDROWS))]

ARRAYS = {
    "rewards": dict(rewards=REWARDS),
    "rewardsElementwise": dict(rewardsElementwise=REWARDROWS),
    "tranMatWithZeros": dict(tranMatWithZeros=DENSE),
    "tranMatElementwise": dict(tranMatElementwise=ROWS),
    "tranMatProbs and tranMatColumns": dict(tranMatProbs=PROBS, tranMatColumns=COLUMNS),
}
LISTS = {
    "rewards": dict(rewards=REWARDS.tolist()),
    "rewardsElementwise": dict(rewardsElementwise=[[int(s), int(a), r] for s, a, r in REWARDROWS.tolist()]),
    "tranMatWithZeros": dict(tranMatWithZeros=DENSE.tolist()),
    "tranMatElementwise": dict(tranMatElementwise=[[int(s), int(a), int(j), p] for s, a, j, p in ROWS.tolist()]),
    "tranMatProbs and tranMatColumns": dict(tranMatProbs=PROBS.tolist(), tranMatColumns=COLUMNS.tolist()),
}
if set(ARRAYS) != set(REWARDFORMATS + TRANSITIONFORMATS) or set(LISTS) != set(ARRAYS):
    sys.exit("Validation test failed at initialization: the formats of the arrays and the lists do not match.")

# The solver settings of case 2: every algorithm with every update method, and
# with standard updates under the average-reward criterion
DISCOUNTEDSETTINGS = [dict(algorithm=algorithm, update=update, criterion="discounted") for algorithm in ALGORITHMS for update in UPDATES]
SETTINGS = DISCOUNTEDSETTINGS + [dict(algorithm=algorithm, update="standard", criterion="average") for algorithm in ALGORITHMS]


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def run(arguments, parallel=False, **setting):
    '''Defines the model by the arguments of mdp() and solves it; returns the policy, the values, the iterations and anything printed.'''
    def solve():
        mdl = mdpsolver.model()
        mdl.mdp(discount=DISCOUNT, **arguments)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mdl.solve(tolerance=TOLERANCE, SORrelaxation=SORRELAXATION, parallel=parallel, **setting)
        return list(mdl.getPolicy()), list(mdl.getValueVector()), mdl.getIterations()
    (policy, values, iterations), output = quietly(solve)
    return (policy, values, iterations), output


def describe(arguments):
    '''A short label for the arguments: each with its type, and an array's dtype and shape.'''
    return ", ".join("%s as %s" % (name, "%s %s array" % (value.dtype, value.shape) if isinstance(value, np.ndarray) else type(value).__name__)
                     for name, value in arguments.items())


def same(case, label, arrays, lists, settings=SETTINGS, reference="the lists"):
    '''Exits unless the arguments with arrays give exactly the results of the arguments with lists (the reference) under every setting, in serial and in parallel.'''
    for setting in settings:
        for parallel in (False, True):
            fromarrays, output = run(arrays, parallel, **setting)
            if output:
                sys.exit("Validation test failed at case %d: %s printed %r." % (case, label, output.strip()))
            fromlists, _ = run(lists, parallel, **setting)
            if fromarrays != fromlists:
                what = [name for name, x, y in zip(("policy", "values", "iterations"), fromarrays, fromlists) if x != y]
                sys.exit("Validation test failed at case %d: %s (%s) under %s, parallel=%s, gives other %s than %s." % (case, label, describe(arrays), setting, parallel, " and ".join(what), reference))


def rejected(case, label, expected, words, arguments):
    '''Exits unless mdp() with the arguments raises the expected error, naming every word in words, and prints nothing besides.'''
    mdl = mdpsolver.model()
    try:
        _, output = quietly(lambda: mdl.mdp(discount=DISCOUNT, **arguments))
    except Exception as error:
        if type(error) is not expected or not all(word in str(error) for word in words):
            sys.exit("Validation test failed at case %d: %s raised %r, where it should raise a %s naming %s." % (case, label, error, expected.__name__, " and ".join(words)))
        return mdl
    sys.exit("Validation test failed at case %d: %s raised no error, where it should raise a %s." % (case, label, expected.__name__))


# ------------------------------------------------------------------
# CASE 1: The exact solution from arrays
# ------------------------------------------------------------------

for algorithm in ALGORITHMS:
    (policy, values, _), output = run(dict(rewards=REWARDS, tranMatProbs=PROBS, tranMatColumns=COLUMNS), algorithm=algorithm)
    if output:
        sys.exit("Validation test failed at case 1: %s printed %r." % (algorithm, output.strip()))
    if policy != EXACTPOLICY:
        sys.exit("Validation test failed at case 1: %s gives a policy that differs from the exact one in %d states." % (algorithm, sum(a != b for a, b in zip(policy, EXACTPOLICY))))
    difference = float(np.max(np.abs(np.array(values) - EXACTVALUES)))
    if difference > TOLERANCE:
        sys.exit("Validation test failed at case 1: %s gives values that differ by %.3e from the exact ones, which is more than %.1e." % (algorithm, difference, TOLERANCE))


# ------------------------------------------------------------------
# CASE 2: Every format as arrays and as lists
# ------------------------------------------------------------------

for rewardformat in REWARDFORMATS:
    for transitionformat in TRANSITIONFORMATS:
        same(2, "%s with %s" % (rewardformat, transitionformat),
             {**ARRAYS[rewardformat], **ARRAYS[transitionformat]}, {**LISTS[rewardformat], **LISTS[transitionformat]})


# ------------------------------------------------------------------
# CASE 3: Other types, layouts, padding, and arrays mixed with lists
# ------------------------------------------------------------------

LISTREFERENCE = {**LISTS["rewards"], **LISTS["tranMatProbs and tranMatColumns"]}
PADDEDPROBS = np.concatenate([PROBS, np.zeros((NSTATES, NACTIONS, 2))], axis=2)
PADDEDCOLUMNS = np.concatenate([COLUMNS, np.repeat(np.arange(NSTATES)[:, None, None], NACTIONS, axis=1).repeat(2, axis=2)], axis=2)
WIDE = np.zeros((NSTATES, NACTIONS, 2 * NNEXT))
WIDE[:, :, ::2] = PROBS
VARIANTS = (
    ("integer rewards", dict(rewards=np.round(REWARDS).astype(np.int32), tranMatProbs=PROBS, tranMatColumns=COLUMNS),
     dict(rewards=np.round(REWARDS).astype(int).tolist(), tranMatProbs=PROBS.tolist(), tranMatColumns=COLUMNS.tolist())),
    ("float32 probabilities", dict(rewards=REWARDS, tranMatProbs=PROBS.astype(np.float32), tranMatColumns=COLUMNS),
     dict(rewards=REWARDS.tolist(), tranMatProbs=PROBS.astype(np.float32).astype(float).tolist(), tranMatColumns=COLUMNS.tolist())),
    ("int32 column indices", dict(rewards=REWARDS, tranMatProbs=PROBS, tranMatColumns=COLUMNS.astype(np.int32)), LISTREFERENCE),
    ("float column indices", dict(rewards=REWARDS, tranMatProbs=PROBS, tranMatColumns=COLUMNS.astype(float)), LISTREFERENCE),
    ("Fortran order", dict(rewards=np.asfortranarray(REWARDS), tranMatProbs=np.asfortranarray(PROBS), tranMatColumns=np.asfortranarray(COLUMNS)), LISTREFERENCE),
    ("views with strides", dict(rewards=REWARDS, tranMatProbs=WIDE[:, :, ::2], tranMatColumns=COLUMNS), LISTREFERENCE),
    ("rows padded with probability 0", dict(rewards=REWARDS, tranMatProbs=PADDEDPROBS, tranMatColumns=PADDEDCOLUMNS), LISTREFERENCE),
    ("rewards as an array, transitions as lists", {**ARRAYS["rewards"], **LISTS["tranMatProbs and tranMatColumns"]}, LISTREFERENCE),
    ("rewards as a list, transitions as an array", {**LISTS["rewards"], **ARRAYS["tranMatElementwise"]}, {**LISTS["rewards"], **LISTS["tranMatElementwise"]}),
    ("tranMatProbs as an array, tranMatColumns as a list", dict(rewards=REWARDS, tranMatProbs=PROBS, tranMatColumns=COLUMNS.tolist()), LISTREFERENCE),
    ("tranMatProbs as a list, tranMatColumns as an array", dict(rewards=REWARDS, tranMatProbs=PROBS.tolist(), tranMatColumns=COLUMNS), LISTREFERENCE),
)
# (under discounting only: these are about reading the input, and with the
# integer rewards the average-reward iterates would grow too large for the tolerance)
for label, arrays, lists in VARIANTS:
    same(3, label, arrays, lists, DISCOUNTEDSETTINGS)


# ------------------------------------------------------------------
# CASE 4: Arrays that do not fit their format
# ------------------------------------------------------------------

RAGGED = [[list(row) for row in state] for state in COLUMNS.tolist()]
RAGGED[0][0] = RAGGED[0][0][:-1]
MALFORMED = (
    # (label, error, words the message must name, arguments, raised before loading)
    ("rewards with one dimension", ValueError, ["rewards", "2 dimensions"], dict(rewards=REWARDS[0], tranMatProbs=PROBS, tranMatColumns=COLUMNS), True),
    ("tranMatWithZeros with two dimensions", ValueError, ["tranMatWithZeros", "3 dimensions"], dict(rewards=REWARDS, tranMatWithZeros=DENSE[0]), True),
    ("a number as rewards", ValueError, ["rewards", "2 dimensions"], dict(rewards=np.float64(1.0), tranMatProbs=PROBS, tranMatColumns=COLUMNS), True),
    ("tranMatProbs and tranMatColumns of different shapes", ValueError, ["tranMatProbs", "tranMatColumns", "shape"], dict(rewards=REWARDS, tranMatProbs=PROBS, tranMatColumns=COLUMNS[:, :, :-1]), True),
    ("a ragged list with an array", ValueError, ["tranMatColumns", "array"], dict(rewards=REWARDS, tranMatProbs=PROBS, tranMatColumns=RAGGED), True),
    ("a fractional column index", ValueError, ["tranMatColumns", "not an integer"], dict(rewards=REWARDS, tranMatProbs=PROBS, tranMatColumns=COLUMNS + 0.5), True),
    ("text", TypeError, ["rewards"], dict(rewards=np.array([["a", "b"]]), tranMatProbs=PROBS, tranMatColumns=COLUMNS), True),
    ("an elementwise array with too few columns", ValueError, ["tranMatElementwise", "columns"], dict(rewards=REWARDS, tranMatElementwise=ROWS[:, :3]), False),
    ("a negative state", ValueError, ["tranMatElementwise[%d]" % len(ROWS), "negative state"], dict(rewards=REWARDS, tranMatElementwise=np.vstack([ROWS, [-1, 0, 0, 1.0]])), False),
    ("a fractional action", ValueError, ["rewardsElementwise[%d]" % len(REWARDROWS), "action", "0.5"], dict(rewardsElementwise=np.vstack([REWARDROWS, [0, 0.5, 1.0]]), tranMatElementwise=ROWS), False),
    ("a next state that is NaN", ValueError, ["tranMatElementwise[%d]" % len(ROWS), "next-state", "nan"], dict(rewards=REWARDS, tranMatElementwise=np.vstack([ROWS, [0, 0, np.nan, 1.0]])), False),
)
for label, expected, words, arguments, beforeloading in MALFORMED:
    mdl = rejected(4, label, expected, words, arguments)
    if beforeloading:
        try:
            _, output = quietly(lambda: mdl.mdp(discount=DISCOUNT, **ARRAYS["rewards"], **ARRAYS["tranMatElementwise"]))
        except Exception as error:
            sys.exit("Validation test failed at case 4: after %s was rejected, a valid mdp() on the same model object raised %r." % (label, error))

# An empty array is not given: in a directory without the default rewards.csv,
# mdp() raises the error for missing rewards
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as emptydir:
    try:
        os.chdir(emptydir)
        rejected(4, "an empty rewards array", FileNotFoundError, ["rewards.csv"], dict(rewards=np.zeros((0, NACTIONS)), tranMatProbs=PROBS, tranMatColumns=COLUMNS))
    finally:
        os.chdir(cwd)


# ------------------------------------------------------------------
# CASE 5: Initial vectors as arrays
# ------------------------------------------------------------------

INITPOLICY = rng.integers(0, NACTIONS, size=NSTATES)
INITVALUES = rng.normal(size=NSTATES) * 100.0
for algorithm in ("pi", "mpi"):
    fromarrays, _ = run(ARRAYS["rewards"] | ARRAYS["tranMatProbs and tranMatColumns"], algorithm=algorithm, initPolicy=INITPOLICY, initValueVector=INITVALUES)
    fromlists, _ = run(ARRAYS["rewards"] | ARRAYS["tranMatProbs and tranMatColumns"], algorithm=algorithm, initPolicy=INITPOLICY.tolist(), initValueVector=INITVALUES.tolist())
    if fromarrays != fromlists:
        sys.exit("Validation test failed at case 5: %s with initPolicy and initValueVector as arrays does not give the results of the same lists." % algorithm)


# ------------------------------------------------------------------
# CASE 6: float32 transition probabilities
# ------------------------------------------------------------------

# The probabilities rounded to float32, and the same numbers in float64
SINGLE = {
    "tranMatWithZeros": dict(tranMatWithZeros=DENSE.astype(np.float32)),
    "tranMatElementwise": dict(tranMatElementwise=ROWS.astype(np.float32)),
    "tranMatProbs and tranMatColumns": dict(tranMatProbs=PROBS.astype(np.float32), tranMatColumns=COLUMNS),
}
if np.any(ROWS[:, :3].astype(np.float32) != ROWS[:, :3]):
    sys.exit("Validation test failed at initialization: the indices of the elementwise rows are not exact in float32.")
for transitionformat, arguments in SINGLE.items():
    double = {name: value.astype(np.float64) if value.dtype == np.float32 else value for name, value in arguments.items()}
    same(6, "%s as float32" % transitionformat, {**ARRAYS["rewards"], **arguments}, {**ARRAYS["rewards"], **double},
         DISCOUNTEDSETTINGS, reference="the same numbers in float64")

def problems(arguments):
    mdl = mdpsolver.model()
    mdl.mdp(discount=DISCOUNT, **ARRAYS["rewards"], **arguments)
    found, output = quietly(lambda: mdl.checkModel())
    return found
FLOAT32 = SINGLE["tranMatProbs and tranMatColumns"]
if problems(FLOAT32):
    sys.exit("Validation test failed at case 6: checkModel() finds %r in a well-formed model with float32 probabilities." % problems(FLOAT32))
if not any("do not sum to 1" in problem for problem in problems(dict(tranMatProbs=FLOAT32["tranMatProbs"].astype(np.float64), tranMatColumns=COLUMNS))):
    sys.exit("Validation test failed at case 6: checkModel() finds no sums off by the float32 rounding when the same numbers are given in float64, so the test cannot tell whether float32 probabilities are kept as such.")
WRONGSUM = FLOAT32["tranMatProbs"].copy()
WRONGSUM[0, 0] *= np.float32(0.9)
found = problems(dict(tranMatProbs=WRONGSUM, tranMatColumns=COLUMNS))
if len(found) != 1 or "do not sum to 1" not in found[0] or "state 0, action 0" not in found[0]:
    sys.exit("Validation test failed at case 6: checkModel() on float32 probabilities with one sum of 0.9 found %r, where it should report that sum alone." % found)


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
