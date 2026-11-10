'''
UNIT TEST FOR THE MODEL IMPORT METHODS OF model.mdp() (the formats in which
the rewards and the transition probabilities of an MDP are given).

mdp() accepts the rewards in three formats and the transition probabilities in
four:

    rewards              2D list, rewards[s][a]
    rewardsElementwise   rows [s, a, reward]
    rewardsFromFile      CSV file with a header line and rows s,a,reward
    tranMatWithZeros     dense 3D list, tranMatWithZeros[s][a][j], zeros included
    tranMatElementwise   sparse rows [s, a, j, probability]
    tranMatProbs and     sparse 3D lists of the non-zero probabilities and of
    tranMatColumns       their next-state (column) indices
    tranMatFromFile      CSV file with a header line and rows s,a,j,probability

Each format is a separate route into the same internal representation (for
every state-action pair, the non-zero probabilities and their columns), so one
MDP given in any combination of the formats must give the same policy and the
same value vector. The test writes a single MDP in every format and requires
every one of the 3 x 4 combinations to reproduce the solution obtained from the
dense formats (rewards and tranMatWithZeros), which serve as the reference. The
list of formats is checked against the arguments of mdp(), so that a format
added later cannot go untested unnoticed.

The MDP is chosen to exercise what a conversion between the formats can get
wrong: The states have different numbers of actions (from one to four), so the
number of actions has to be inferred state by state from the elementwise rows;
there are more than ten states, so the files hold indices of two digits; and
there are deterministic transitions, transitions back into the same state
(which the Gauss-Seidel and SOR updates handle apart from the others), a zero
reward, and a probability small enough to be written to file in exponent
notation.

Every comparison is repeated under several solver settings, which together
cover both optimality criteria, all three algorithms and all three
value-update methods, so that a difference between the formats cannot hide
behind a single way of reading the matrix.

Sub-tests:
    Case 1: The reference reproduces the exact solution of the MDP, computed by
            policy iteration with numpy, so that the formats are compared on
            the model the test defines rather than on one misread alike by all.
    Case 2: Every combination of a reward format and a transition format gives
            the policy and the value vector of the reference.
    Case 3: The order of the rows does not matter: The elementwise lists and
            the files with their rows shuffled, and tranMatProbs and
            tranMatColumns with the entries of each state-action pair permuted
            alike, give the results of the reference.
    Case 4: Files with Windows line endings (CR LF), and files without a line
            break after the last row, give the results of the reference.

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import inspect
import itertools
import os
import random
import sys
import tempfile

import numpy as np

from loadmdpsolver import mdpsolver, finalline


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

DISCOUNT = 0.9
TOLERANCE = 1e-9   # the epsilon handed to solve()
SEED = 0           # for the shuffles of case 3

# The formats hold the same numbers in the same order, so they give identical
# results here; shuffled rows change only the order of summation, which moves
# the values by rounding (about 1e-15 relative to the largest value). TOL is
# the relative difference allowed between two value vectors.
TOL = 1e-12

# The optimal action of every state with more than one action must beat the
# runner-up by at least MINGAP, or a comparison of policies would hinge on
# rounding rather than on the import (the smallest gap is 2.9 here).
MINGAP = 1e-3

# The MDP: MODEL[s][a] is the pair (reward, {next state: probability}) of
# action a in state s, with the next states in increasing order, which is the
# order the dense format implies.
MODEL = [
    # state 0
    [(2.5, {6: 0.056, 8: 0.244, 9: 0.7}),
     (-9.89, {1: 0.279, 4: 0.024, 8: 0.038, 9: 0.659})],
    # state 1
    [(-1.1, {5: 0.509, 6: 0.491}),
     (5.85, {3: 0.16, 6: 0.452, 9: 0.244, 11: 0.144}),
     (-9.29, {6: 1.0})],
    # state 2: a single action, a zero reward, and a certain move to the last state
    [(0.0, {11: 1.0})],
    # state 3: deterministic actions only
    [(3.84, {10: 1.0}),
     (-2.61, {5: 1.0}),
     (6.6, {7: 1.0}),
     (-4.65, {11: 1.0})],
    # state 4
    [(0.2, {7: 0.092, 8: 0.152, 9: 0.297, 11: 0.459}),
     (7.43, {3: 0.388, 7: 0.271, 11: 0.341})],
    # state 5: the optimal action returns to the state itself
    [(-7.0, {4: 0.38, 9: 0.62}),
     (9.57, {5: 0.35, 8: 0.4, 11: 0.25}),
     (3.53, {1: 0.053, 6: 0.187, 9: 0.76})],
    # state 6
    [(-8.07, {10: 0.008, 11: 0.992}),
     (-3.99, {1: 0.132, 4: 0.713, 9: 0.155})],
    # state 7: a single action
    [(8.9, {6: 0.193, 9: 0.369, 10: 0.438})],
    # state 8
    [(1.05, {0: 0.061, 1: 0.315, 4: 0.192, 10: 0.432}),
     (-5.21, {0: 0.548, 10: 0.212, 11: 0.24}),
     (5.03, {0: 0.372, 11: 0.628})],
    # state 9: the optimal action almost surely returns to the state itself,
    # and its other probability is written as 2e-05 in the files
    [(-9.39, {1: 0.079, 4: 0.354, 10: 0.09, 11: 0.477}),
     (1.81, {3: 0.356, 8: 0.644}),
     (0.38, {7: 0.412, 8: 0.521, 10: 0.067}),
     (5.06, {9: 0.99998, 10: 2e-05})],
    # state 10
    [(-7.26, {4: 0.485, 7: 0.142, 9: 0.166, 11: 0.207}),
     (4.52, {2: 1.0})],
    # state 11
    [(-6.03, {1: 0.573, 3: 0.291, 4: 0.122, 10: 0.014}),
     (9.04, {2: 0.98, 5: 0.02}),
     (0.31, {4: 0.426, 8: 0.403, 9: 0.116, 11: 0.055})],
]

NSTATES = len(MODEL)

# The solver settings under which every comparison is made. Parallel computing
# is only available with standard updates (see the wiki), so the Gauss-Seidel
# and SOR settings run serially, as in test1.py. The two settings with standard
# updates and the discounted criterion cover the parallel and the serial code.
SOLVERSETTINGS = (
    ("discounted MPI with standard updates",
     dict(criterion="discounted", algorithm="mpi", update="standard", parallel=True)),
    ("discounted VI with standard updates",
     dict(criterion="discounted", algorithm="vi", update="standard", parallel=False)),
    ("discounted PI with Gauss-Seidel updates",
     dict(criterion="discounted", algorithm="pi", update="gs", parallel=False)),
    ("discounted VI with SOR updates",
     dict(criterion="discounted", algorithm="vi", update="sor", SORrelaxation=1.2, parallel=False)),
    ("average-reward MPI with standard updates",
     dict(criterion="average", algorithm="mpi", update="standard", parallel=True)),
)


# The model above is the whole test, so a typo in it would quietly make a case
# meaningless rather than fail it. It is checked to be an MDP, and to carry
# each of the features the test relies on.
for s, state in enumerate(MODEL):
    if len(state) == 0:
        sys.exit("Validation test failed at initialization: state %d has no actions." % s)

    for a, (reward, probs) in enumerate(state):
        if any(j < 0 or j >= NSTATES for j in probs):
            sys.exit("Validation test failed at initialization: action %d in state %d leads to a state outside the model." % (a, s))

        if any(not 0.0 < p <= 1.0 for p in probs.values()):
            sys.exit("Validation test failed at initialization: action %d in state %d has a probability outside (0, 1]." % (a, s))

        if abs(sum(probs.values()) - 1.0) > 1e-12:
            sys.exit("Validation test failed at initialization: the probabilities of action %d in state %d sum to %.15f, not 1." % (a, s, sum(probs.values())))

        if list(probs) != sorted(probs):
            sys.exit("Validation test failed at initialization: the next states of action %d in state %d are not in increasing order." % (a, s))

actioncounts = [len(state) for state in MODEL]
allprobs = [(s, j, p) for s, state in enumerate(MODEL) for _, probs in state for j, p in probs.items()]

if min(actioncounts) == max(actioncounts) or min(actioncounts) != 1:
    sys.exit("Validation test failed at initialization: the states do not have different numbers of actions down to one, so the case for unequal action sets is not being run.")

if NSTATES <= 10:
    sys.exit("Validation test failed at initialization: the model has no state index of two digits, so the files are not tested on one.")

if not any(len(probs) == 1 for state in MODEL for _, probs in state):
    sys.exit("Validation test failed at initialization: the model has no deterministic transition.")

if not any(s == j for s, j, _ in allprobs):
    sys.exit("Validation test failed at initialization: the model has no transition back into the same state, which Gauss-Seidel and SOR handle apart.")

if not any("e" in repr(p) for _, _, p in allprobs):
    sys.exit("Validation test failed at initialization: no probability is written in exponent notation, so the files are not tested on one.")

if not any(reward == 0.0 for state in MODEL for reward, _ in state):
    sys.exit("Validation test failed at initialization: the model has no zero reward.")

if len(allprobs) >= NSTATES * sum(actioncounts):
    sys.exit("Validation test failed at initialization: the dense transition matrix has no zeros, so the sparse formats leave nothing out.")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def denserewards(model):
    '''The rewards as a 2D list, rewards[s][a].'''
    return [[float(reward) for reward, _ in state] for state in model]


def elementwiserewards(model):
    '''The rewards as rows [s, a, reward].'''
    return [[s, a, float(reward)]
            for s, state in enumerate(model)
            for a, (reward, _) in enumerate(state)]


def densetranmat(model):
    '''The transition probabilities as a 3D list with zeros, P[s][a][j].'''
    nstates = len(model)
    return [[[float(probs.get(j, 0.0)) for j in range(nstates)]
             for _, probs in state]
            for state in model]


def elementwisetranmat(model):
    '''The non-zero transition probabilities as rows [s, a, j, probability].'''
    return [[s, a, j, float(p)]
            for s, state in enumerate(model)
            for a, (_, probs) in enumerate(state)
            for j, p in probs.items()]


def probsandcolumns(model):
    '''The non-zero transition probabilities and their columns as two 3D lists.'''
    probs = [[[float(p) for p in jumps.values()] for _, jumps in state] for state in model]
    columns = [[list(jumps.keys()) for _, jumps in state] for state in model]
    return probs, columns


def writefile(path, header, rows, newline="\n", lastnewline=True):
    '''
    Writes rows to a comma-separated file with a header line. Numbers are
    written with repr(), which round-trips a float exactly, so the file holds
    the same numbers as the lists. The file is opened with newline="", so that
    the line endings are the ones asked for on every platform.
    '''
    lines = [header] + [",".join(repr(x) for x in row) for row in rows]
    text = newline.join(lines) + (newline if lastnewline else "")
    with open(path, "w", newline="") as file:
        file.write(text)
    return path


def solve(arguments, setting):
    '''Solves the MDP given by the mdp() arguments; returns the policy and the value vector.'''
    mdl = mdpsolver.model()
    mdl.mdp(discount=DISCOUNT, **arguments)
    mdl.solve(tolerance=TOLERANCE, **setting)
    return list(mdl.getPolicy()), np.array(mdl.getValueVector(), dtype=float)


def compare(case, label, settinglabel, result, reference):
    '''Ends the test if a result differs from the reference.'''
    policy, values = result
    refpolicy, refvalues = reference

    if policy != refpolicy:
        sys.exit("Validation test failed at case %d: %s gives the policy %s under %s, where the reference gives %s." % (case, label, policy, settinglabel, refpolicy))

    if values.shape != refvalues.shape:
        sys.exit("Validation test failed at case %d: %s gives %d values under %s, where the reference gives %d." % (case, label, values.size, settinglabel, refvalues.size))

    allowed = TOL * float(np.max(np.abs(refvalues)))
    difference = float(np.max(np.abs(values - refvalues)))

    if not np.all(np.isfinite(values)) or difference > allowed:
        sys.exit("Validation test failed at case %d: %s gives values that differ by %.3e from those of the reference under %s, which is more than %.1e." % (case, label, difference, settinglabel, allowed))


def exactsolution(model, discount):
    '''
    The optimal policy and value vector of the discounted MDP by policy
    iteration, each policy evaluated exactly by solving the linear system;
    also the gap between the best and the second-best action of every state
    with more than one action.
    '''
    rewards = denserewards(model)
    tranmat = [[np.array(row) for row in state] for state in densetranmat(model)]
    nstates = len(model)
    policy = [0] * nstates

    while True:
        ppolicy = np.array([tranmat[s][policy[s]] for s in range(nstates)])
        rpolicy = np.array([rewards[s][policy[s]] for s in range(nstates)])
        values = np.linalg.solve(np.eye(nstates) - discount * ppolicy, rpolicy)
        qvalues = [[rewards[s][a] + discount * float(tranmat[s][a] @ values)
                    for a in range(len(rewards[s]))]
                   for s in range(nstates)]
        improved = [int(np.argmax(q)) for q in qvalues]
        if improved == policy:
            gaps = [sorted(q)[-1] - sorted(q)[-2] for q in qvalues if len(q) > 1]
            return policy, values, gaps
        policy = improved


# ------------------------------------------------------------------
# THE MODEL IN EVERY FORMAT
# ------------------------------------------------------------------

# The files are written to a temporary directory rather than next to the test.
# It is removed at the end, and also when the test ends at a failed case, since
# sys.exit() lets the interpreter exit normally and run the directory's
# finalizer.
TMPDIR = tempfile.TemporaryDirectory()

REWARDHEADER = "state,action,reward"
TRANMATHEADER = "stateFrom,action,stateTo,probability"

tranmatprobs, tranmatcolumns = probsandcolumns(MODEL)

REWARDFORMATS = {
    "rewards": {"rewards": denserewards(MODEL)},
    "rewardsElementwise": {"rewardsElementwise": elementwiserewards(MODEL)},
    "rewardsFromFile": {"rewardsFromFile": writefile(
        os.path.join(TMPDIR.name, "rewards.csv"), REWARDHEADER, elementwiserewards(MODEL))},
}

TRANMATFORMATS = {
    "tranMatWithZeros": {"tranMatWithZeros": densetranmat(MODEL)},
    "tranMatElementwise": {"tranMatElementwise": elementwisetranmat(MODEL)},
    "tranMatProbs and tranMatColumns": {"tranMatProbs": tranmatprobs,
                                        "tranMatColumns": tranmatcolumns},
    "tranMatFromFile": {"tranMatFromFile": writefile(
        os.path.join(TMPDIR.name, "transitions.csv"), TRANMATHEADER, elementwisetranmat(MODEL))},
}

REFERENCEARGUMENTS = {**REWARDFORMATS["rewards"], **TRANMATFORMATS["tranMatWithZeros"]}

# Every import argument of mdp() must be covered by a format
covered = set()
for arguments in list(REWARDFORMATS.values()) + list(TRANMATFORMATS.values()):
    covered.update(arguments)

importarguments = set(inspect.signature(mdpsolver.model.mdp).parameters) - {"self", "discount"}

if covered != importarguments:
    sys.exit("Validation test failed at initialization: mdp() takes the import arguments %s, but the test covers %s." % (sorted(importarguments), sorted(covered)))


# ------------------------------------------------------------------
# CASE 1: The reference against the exact solution
# ------------------------------------------------------------------

# Under the average-reward criterion the value vector is determined only up to
# a constant, so only the discounted settings are compared with the exact
# solution here. The average-reward setting is still compared between the
# formats in the cases below. The solution of solve() is epsilon-optimal for
# the tolerance it is handed, so the values of the reference have to lie within
# TOLERANCE of the exact ones (they lie within 3e-10 here).
exactpolicy, exactvalues, gaps = exactsolution(MODEL, DISCOUNT)

if min(gaps) < MINGAP:
    sys.exit("Validation test failed at case 1: the best action of a state beats the runner-up by only %.3e, so the comparisons of policies would hinge on rounding." % min(gaps))

reference = {}
for settinglabel, setting in SOLVERSETTINGS:
    reference[settinglabel] = solve(REFERENCEARGUMENTS, setting)

    if setting["criterion"] != "discounted":
        continue

    policy, values = reference[settinglabel]

    if policy != exactpolicy:
        sys.exit("Validation test failed at case 1: the reference gives the policy %s under %s, where the exact solution is %s." % (policy, settinglabel, exactpolicy))

    if values.shape != exactvalues.shape:
        sys.exit("Validation test failed at case 1: the reference gives %d values under %s, for a model of %d states." % (values.size, settinglabel, NSTATES))

    difference = float(np.max(np.abs(values - exactvalues)))

    if not np.all(np.isfinite(values)) or difference > TOLERANCE:
        sys.exit("Validation test failed at case 1: the values of the reference differ by %.3e from the exact ones under %s, which is more than the tolerance of %.1e handed to solve()." % (difference, settinglabel, TOLERANCE))


# ------------------------------------------------------------------
# CASE 2: Every combination of formats
# ------------------------------------------------------------------

for settinglabel, setting in SOLVERSETTINGS:
    for (rewardlabel, rewardarguments), (tranmatlabel, tranmatarguments) in itertools.product(
            REWARDFORMATS.items(), TRANMATFORMATS.items()):
        result = solve({**rewardarguments, **tranmatarguments}, setting)
        compare(2, "%s with %s" % (rewardlabel, tranmatlabel), settinglabel, result, reference[settinglabel])


# ------------------------------------------------------------------
# CASE 3: The order of the rows
# ------------------------------------------------------------------

# Nothing in the documentation asks for the rows to be sorted, and the loaders
# place each row by its indices rather than by its position, so shuffled rows
# must describe the same MDP. Within a state-action pair, the order of the
# entries changes the order in which the solver sums them, so the values may
# move by rounding (TOL), but no more.
rng = random.Random(SEED)

shuffledrewards = elementwiserewards(MODEL)
rng.shuffle(shuffledrewards)

shuffledtranmat = elementwisetranmat(MODEL)
rng.shuffle(shuffledtranmat)

permutedprobs, permutedcolumns = probsandcolumns(MODEL)
for s in range(NSTATES):
    for a in range(len(MODEL[s])):
        order = list(range(len(permutedprobs[s][a])))
        rng.shuffle(order)
        permutedprobs[s][a] = [permutedprobs[s][a][i] for i in order]
        permutedcolumns[s][a] = [permutedcolumns[s][a][i] for i in order]

# the case is only worth running if the order has actually changed
if shuffledrewards == elementwiserewards(MODEL) or shuffledtranmat == elementwisetranmat(MODEL):
    sys.exit("Validation test failed at case 3: a shuffle left the elementwise rows in their original order.")

if permutedcolumns == tranmatcolumns:
    sys.exit("Validation test failed at case 3: the permutation left every list of columns in its original order.")

SHUFFLEDREWARDFORMATS = {
    "rewards": REWARDFORMATS["rewards"],
    "rewardsElementwise (shuffled)": {"rewardsElementwise": shuffledrewards},
    "rewardsFromFile (shuffled)": {"rewardsFromFile": writefile(
        os.path.join(TMPDIR.name, "rewards_shuffled.csv"), REWARDHEADER, shuffledrewards)},
}

SHUFFLEDTRANMATFORMATS = {
    "tranMatWithZeros": TRANMATFORMATS["tranMatWithZeros"],
    "tranMatElementwise (shuffled)": {"tranMatElementwise": shuffledtranmat},
    "tranMatProbs and tranMatColumns (permuted)": {"tranMatProbs": permutedprobs,
                                                   "tranMatColumns": permutedcolumns},
    "tranMatFromFile (shuffled)": {"tranMatFromFile": writefile(
        os.path.join(TMPDIR.name, "transitions_shuffled.csv"), TRANMATHEADER, shuffledtranmat)},
}

for settinglabel, setting in SOLVERSETTINGS:
    for (rewardlabel, rewardarguments), (tranmatlabel, tranmatarguments) in itertools.product(
            SHUFFLEDREWARDFORMATS.items(), SHUFFLEDTRANMATFORMATS.items()):
        result = solve({**rewardarguments, **tranmatarguments}, setting)
        compare(3, "%s with %s" % (rewardlabel, tranmatlabel), settinglabel, result, reference[settinglabel])


# ------------------------------------------------------------------
# CASE 4: Line endings in the files
# ------------------------------------------------------------------

# Python's text mode writes CR LF line endings on Windows, so a file written
# there without newline="" ends every line, the header included, in a carriage
# return, which the reader has to tolerate. A file whose last row has no line
# break after it is just as common.
FILEVARIANTS = (
    ("files with Windows line endings", "crlf", "\r\n", True),
    ("files without a line break after the last row", "nolastbreak", "\n", False),
)

for variantlabel, suffix, newline, lastnewline in FILEVARIANTS:
    rewardfile = writefile(os.path.join(TMPDIR.name, "rewards_%s.csv" % suffix), REWARDHEADER,
                           elementwiserewards(MODEL), newline, lastnewline)
    tranmatfile = writefile(os.path.join(TMPDIR.name, "transitions_%s.csv" % suffix), TRANMATHEADER,
                            elementwisetranmat(MODEL), newline, lastnewline)

    with open(tranmatfile, "rb") as file:
        content = file.read()

    # the case is only worth running if the file has the ending it is named for
    if content.endswith(b"\n") != lastnewline or (b"\r\n" in content) != (newline == "\r\n"):
        sys.exit("Validation test failed at case 4: the %s were not written as intended." % variantlabel)

    for settinglabel, setting in SOLVERSETTINGS:
        result = solve({"rewardsFromFile": rewardfile, "tranMatFromFile": tranmatfile}, setting)
        compare(4, "rewardsFromFile with tranMatFromFile, as %s," % variantlabel, settinglabel, result, reference[settinglabel])


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

TMPDIR.cleanup()

print(finalline(__file__))
