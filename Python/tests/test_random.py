'''
UNIT TEST COMPARING VALUE ITERATION, POLICY ITERATION AND MODIFIED POLICY
ITERATION ON NPROBLEMS SMALL RANDOM PROBLEMS.

Each problem has from MINSTATES to MAXSTATES states with from 1 to MAXACTIONS
actions each, sparse transition probabilities, and a discount factor drawn
from DISCOUNTRANGE. The reward of an action is a level for its state, drawn
uniformly from [-10, 10], plus up to ACTIONSPREAD either way for the action, so
that where an action leads matters as much as what it pays (with rewards drawn
freely from [-10, 10], two thirds of the problems would be solved by taking
the best immediate reward). Every action
leads back to state 0 with positive probability, besides to up to MAXEXTRA other
states, so every policy is unichain (state 0 is reached from every state) and
aperiodic (state 0 can follow itself), which value iteration needs under the
average-reward criterion (Puterman, 1994, Section 8.5).

The exact solution of every problem is computed by policy iteration with exact
evaluation (knownsolutions.py), and a problem is kept only if the optimal action
of every state beats the runner-up by MINGAP under both criteria, so that the
comparison of policies cannot hinge on rounding; a problem that fails is
replaced by the next one drawn. The problems are drawn with Python's random
module from a fixed seed, using only random(), whose sequence is the same in
every version of Python, so every run tests the same problems.

Sub-tests:
    Case 1: Discounted reward: value iteration, policy iteration and modified
            policy iteration, each with standard, Gauss-Seidel and SOR updates,
            all give the exact optimal policy of every problem, and so the same
            policy, and values within the tolerance handed to solve() of the
            exact ones.
    Case 2: Average reward: the three algorithms, with standard updates (the
            only ones available under this criterion), all give the exact
            gain-optimal policy of every problem, and values whose differences
            v(s) - v(0) are the bias. (MDPSolver returns iterates that grow by
            the gain in every iteration, so only their differences are known.)
    Case 3: Runtime per iteration: getRuntime() divided by getIterations(),
            averaged over the problems, is shortest for value iteration, whose
            iterations are single updates, and longest for policy iteration,
            which evaluates every policy until the values settle, with modified
            policy iteration in between; under each update method and both
            criteria. This shows which algorithm ran, which cases 1 and 2
            cannot, since all three give the same results.

The comparison is only worth making if the optimal policy is more than taking
the best immediate reward in every state (where value iteration starts from),
which is checked over the problems. Nothing may be printed by any of the runs.

In case 3, modified policy iteration is limited to MPILIMIT evaluation updates
per iteration. With the default limit of 100, its evaluations end early on these
small problems, when the values have settled, just as those of policy iteration
do. The two algorithms then do the same work, and the order in which they run
decides which looks slower.

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.

References:
    Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic
    Dynamic Programming. Wiley. https://doi.org/10.1002/9780470316887
'''

import random
import sys

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import (checkoptimality, elementwisetranmat, exactaverage, exactdiscounted,
                            smallestgap)


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

NPROBLEMS = 30
SEED = 2026
MAXDRAWS = 1000            # draws allowed to find the problems

MINSTATES = 2
MAXSTATES = 8
MAXACTIONS = 4
MAXEXTRA = 3               # successors besides state 0, at most
ACTIONSPREAD = 2.0         # the rewards of a state's actions differ by up to twice this
DISCOUNTRANGE = (0.3, 0.97)

TOLERANCE = 1e-9           # the epsilon handed to solve()

# The span stopping rule of the average-reward criterion bounds the error of
# the gain rather than of the bias, so the bias is compared at a tolerance set
# by measurement: the differences lie within 1e-8 of it here.
BIASTOL = 1e-7

# The optimal action of every state with more than one action must beat the
# runner-up by at least MINGAP, far more than the tolerance
MINGAP = 1e-3

# At least this many of the problems must have an optimal policy other than
# taking the best immediate reward, under each criterion (22 and 24 do here)
MINNONMYOPIC = NPROBLEMS // 2

ALGORITHMS = (("value iteration", "vi"), ("policy iteration", "pi"), ("modified policy iteration", "mpi"))
DISCOUNTEDUPDATES = (("standard updates", dict(update="standard")),
                     ("Gauss-Seidel updates", dict(update="gs")),
                     ("SOR updates with relaxation 1.2", dict(update="sor", SORrelaxation=1.2)))

# The discount factor in mdp() is not used under the average-reward criterion,
# but has to be valid
AVERAGEDISCOUNT = 0.5

# Case 3: the evaluation updates allowed per iteration of modified policy
# iteration (policy iteration uses 1 to 219 here, 19 at the median), and the runs
# timed per solve, of which the fastest counts, since anything else running on
# the computer can only add to a runtime. Measured margins: modified policy
# iteration takes 2.9 to 4.9 times as long per iteration as value iteration, and
# policy iteration 3.7 to 6.2 times as long as modified policy iteration.
MPILIMIT = 5
TIMINGREPEATS = 3
ITERATIONLIMIT = 1000000   # the solver stops here; reaching it means it did not converge
CRITERIA = tuple((label, setting, "discounted") for label, setting in DISCOUNTEDUPDATES) + \
    (("the average-reward criterion", dict(update="standard"), "average"),)


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def drawproblem(rng):
    '''A random problem: its rewards, transitions and discount factor, drawn with rng.random() only.'''
    n = MINSTATES + int(rng.random() * (MAXSTATES - MINSTATES + 1))
    rewards, transitions = [], []
    for s in range(n):
        nactions = 1 + int(rng.random() * MAXACTIONS)
        level = 20.0 * rng.random() - 10.0
        rewards.append([round(level + ACTIONSPREAD * (2.0 * rng.random() - 1.0), 2) for _ in range(nactions)])
        state = []
        for _ in range(nactions):
            nextra = int(rng.random() * (MAXEXTRA + 1))
            successors = sorted({0} | {int(rng.random() * n) for _ in range(nextra)})
            weights = [0.05 + rng.random() for _ in successors]
            total = sum(weights)
            state.append({j: w / total for j, w in zip(successors, weights)})
        transitions.append(state)
    discount = round(DISCOUNTRANGE[0] + (DISCOUNTRANGE[1] - DISCOUNTRANGE[0]) * rng.random(), 3)
    return rewards, transitions, discount


def solve(rewards, transitions, discount, algorithm, setting, criterion="discounted"):
    '''One algorithm under one setting; returns the policy, the value vector and anything printed.'''
    def run():
        mdl = mdpsolver.model()
        mdl.mdp(discount=discount, rewards=rewards, tranMatElementwise=elementwisetranmat(transitions))
        mdl.solve(algorithm=algorithm, tolerance=TOLERANCE, criterion=criterion, **setting)
        return list(mdl.getPolicy()), np.array(mdl.getValueVector(), dtype=float)

    (policy, values), output = quietly(run)
    return policy, values, output


def timeperiteration(rewards, transitions, discount, algorithm, setting, criterion):
    '''The runtime per iteration in milliseconds, over the fastest of TIMINGREPEATS runs, and the number of iterations.'''
    runtimes, iterations = [], set()
    for _ in range(TIMINGREPEATS):
        mdl = mdpsolver.model()
        mdl.mdp(discount=discount, rewards=rewards, tranMatElementwise=elementwisetranmat(transitions))
        mdl.solve(algorithm=algorithm, tolerance=TOLERANCE, criterion=criterion, **setting)
        runtimes.append(mdl.getRuntime())
        iterations.add(mdl.getIterations())
    if len(iterations) != 1:
        return None, sorted(iterations)
    iterations = iterations.pop()
    return (min(runtimes) / iterations if iterations > 0 else None), iterations


def myopic(rewards):
    '''The policy that takes the best immediate reward in every state.'''
    return [int(np.argmax(state)) for state in rewards]


# ------------------------------------------------------------------
# THE PROBLEMS AND THEIR EXACT SOLUTIONS
# ------------------------------------------------------------------

rng = random.Random(SEED)
problems = []
discarded = 0
while len(problems) < NPROBLEMS:
    if len(problems) + discarded >= MAXDRAWS:
        sys.exit("Validation test failed at initialization: only %d of %d draws gave a problem whose optimal actions win by %.0e." % (len(problems), MAXDRAWS, MINGAP))

    rewards, transitions, discount = drawproblem(rng)
    policy, values = exactdiscounted(rewards, transitions, discount)
    averagepolicy, gain, bias = exactaverage(rewards, transitions)

    if smallestgap(rewards, transitions, values, discount) < MINGAP or \
            smallestgap(rewards, transitions, bias, 1.0) < MINGAP:
        discarded += 1
        continue
    problems.append(dict(rewards=rewards, transitions=transitions, discount=discount,
                         policy=policy, values=values,
                         averagepolicy=averagepolicy, gain=gain, bias=bias))

# The exact solutions are the whole test, so they are checked against the
# optimality equations, and the problems for the features the cases rely on,
# before they are used.
for k, problem in enumerate(problems):
    name = "random problem %d" % k
    checkoptimality(name + " at discount %g" % problem["discount"], problem["rewards"], problem["transitions"],
                    problem["policy"], problem["values"], problem["discount"], MINGAP)
    checkoptimality(name + " under the average-reward criterion", problem["rewards"], problem["transitions"],
                    problem["averagepolicy"], problem["bias"], 1.0, MINGAP, problem["gain"])

    for s, state in enumerate(problem["transitions"]):
        for a, probs in enumerate(state):
            if not probs.get(0, 0.0) > 0.0 or abs(sum(probs.values()) - 1.0) > 1e-12:
                sys.exit("Validation test failed at initialization: action %d in state %d of %s does not lead back to state 0, or its probabilities do not sum to 1." % (a, s, name))

sizes = [len(problem["rewards"]) for problem in problems]
if min(sizes) > MINSTATES or max(sizes) < MAXSTATES:
    sys.exit("Validation test failed at initialization: the problems do not span the sizes from %d to %d states (they have %d to %d)." % (MINSTATES, MAXSTATES, min(sizes), max(sizes)))

nonmyopic = sum(problem["policy"] != myopic(problem["rewards"]) for problem in problems)
averagenonmyopic = sum(problem["averagepolicy"] != myopic(problem["rewards"]) for problem in problems)
if nonmyopic < MINNONMYOPIC or averagenonmyopic < MINNONMYOPIC:
    sys.exit("Validation test failed at initialization: only %d (discounted) and %d (average reward) of the problems have an optimal policy other than taking the best immediate reward, so the comparison would test little." % (nonmyopic, averagenonmyopic))


# ------------------------------------------------------------------
# CASE 1: Discounted reward
# ------------------------------------------------------------------

for k, problem in enumerate(problems):
    description = "problem %d (%d states, discount %g)" % (k, len(problem["rewards"]), problem["discount"])
    for algorithmlabel, algorithm in ALGORITHMS:
        for updatelabel, setting in DISCOUNTEDUPDATES:
            label = "in %s, %s with %s" % (description, algorithmlabel, updatelabel)
            policy, values, output = solve(problem["rewards"], problem["transitions"], problem["discount"], algorithm, setting)

            if output:
                sys.exit("Validation test failed at case 1: %s printed %r." % (label, output.strip()))
            if policy != problem["policy"]:
                sys.exit("Validation test failed at case 1: %s gives the policy %s, where the exact one is %s." % (label, policy, problem["policy"]))
            if values.shape != problem["values"].shape or not np.all(np.isfinite(values)):
                sys.exit("Validation test failed at case 1: %s gives the values %s." % (label, values))
            difference = float(np.max(np.abs(values - problem["values"])))
            if difference > TOLERANCE:
                sys.exit("Validation test failed at case 1: %s gives values that differ by %.3e from the exact ones, which is more than %.1e." % (label, difference, TOLERANCE))


# ------------------------------------------------------------------
# CASE 2: Average reward
# ------------------------------------------------------------------

for k, problem in enumerate(problems):
    description = "problem %d (%d states)" % (k, len(problem["rewards"]))
    for algorithmlabel, algorithm in ALGORITHMS:
        label = "in %s under the average-reward criterion, %s" % (description, algorithmlabel)
        policy, values, output = solve(problem["rewards"], problem["transitions"], AVERAGEDISCOUNT, algorithm,
                                       dict(update="standard"), criterion="average")

        if output:
            sys.exit("Validation test failed at case 2: %s printed %r." % (label, output.strip()))
        if policy != problem["averagepolicy"]:
            sys.exit("Validation test failed at case 2: %s gives the policy %s, where the exact one is %s." % (label, policy, problem["averagepolicy"]))
        if values.shape != problem["bias"].shape or not np.all(np.isfinite(values)):
            sys.exit("Validation test failed at case 2: %s gives the values %s." % (label, values))
        difference = float(np.max(np.abs((values - values[0]) - problem["bias"])))
        if difference > BIASTOL:
            sys.exit("Validation test failed at case 2: %s gives values whose differences from the first differ by %.3e from the bias, which is more than %.1e." % (label, difference, BIASTOL))


# ------------------------------------------------------------------
# CASE 3: Runtime per iteration
# ------------------------------------------------------------------

def timeall():
    '''The runtimes per iteration of the three algorithms on every problem, under every criterion and update method.'''
    times = {}
    for criterionlabel, setting, criterion in CRITERIA:
        for k, problem in enumerate(problems):
            discount = problem["discount"] if criterion == "discounted" else AVERAGEDISCOUNT
            for algorithmlabel, algorithm in ALGORITHMS:
                runsetting = dict(setting, parIterLim=MPILIMIT) if algorithm == "mpi" else setting
                times[criterionlabel, k, algorithm] = timeperiteration(problem["rewards"], problem["transitions"],
                                                                       discount, algorithm, runsetting, criterion)
    return times

times, output = quietly(timeall)
if output:
    sys.exit("Validation test failed at case 3: the timed runs printed %r." % output.strip())

meantimes = {}
for criterionlabel, _, _ in CRITERIA:
    for algorithmlabel, algorithm in ALGORITHMS:
        for k in range(len(problems)):
            runtime, iterations = times[criterionlabel, k, algorithm]
            label = "in problem %d under %s, %s" % (k, criterionlabel, algorithmlabel)
            if isinstance(iterations, list):
                sys.exit("Validation test failed at case 3: %s reports different numbers of iterations in repeated runs: %s." % (label, iterations))
            if not 0 < iterations < ITERATIONLIMIT:
                sys.exit("Validation test failed at case 3: %s reports %d iterations, where 1 to %d are possible before the iteration limit." % (label, iterations, ITERATIONLIMIT - 1))
            # A runtime of 0 is possible: the shortest runs here take about 0.1
            # microseconds, the resolution of the clock on Windows
            if not runtime >= 0.0:
                sys.exit("Validation test failed at case 3: %s reports the runtime %r per iteration." % (label, runtime))
        meantimes[criterionlabel, algorithm] = sum(times[criterionlabel, k, algorithm][0] for k in range(len(problems))) / len(problems)

    vi, pi, mpi = (meantimes[criterionlabel, algorithm] for algorithm in ("vi", "pi", "mpi"))
    summary = "value iteration %.2e ms, modified policy iteration (parIterLim=%d) %.2e ms and policy iteration %.2e ms" % (vi, MPILIMIT, mpi, pi)
    if not vi < mpi < pi:
        sys.exit("Validation test failed at case 3: under %s, the mean runtimes per iteration are %s, where they should rise in that order." % (criterionlabel, summary))


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
