'''
UNIT TEST ON A LARGE MODEL: THE QUEUEING PROBLEM OF THE FORMER test2.py, UNDER
EVERY ALGORITHM AND UPDATE METHOD.

The problem: a loss system with CAPACITY servers, each serving at rate
SERVICERATE, and no room to wait. Customers of two types arrive at rates
ARRIVALRATE1 and ARRIVALRATE2, and are observed as they arrive; each can be
accepted while a server is idle, for a reward of REWARD1 or REWARD2, or
rejected, for nothing. The common type pays little, the rare one much. The
state is the number k of busy servers and the type of the arriving customer:
state k for the first type and CAPACITY + 1 + k for the second, 202 states with
41204 non-zero transition probabilities. After the decision m servers are busy
(m = k, or k + 1 when accepting), and the next arrival finds j of them still
busy with probability
    prod_{i=j+1}^{m} i mu/(i mu + lambda) * lambda/(j mu + lambda),
where lambda is the total arrival rate: each of the busy servers in turn either
finishes before the next arrival or not. (The former test2.py computed the
same probabilities by iterating a chain until less than 1e-9 of the probability
was left; the product is exact, so every row sums to 1, which the
average-reward criterion needs.)

The exact solutions under both criteria are computed by policy iteration with
exact evaluation (knownsolutions.py), and checked against the optimality
equations, with a margin for the optimal actions, before they are used. The
mean of the discounted values rounds to 17075, the value the former test2.py
checked, and the optimal policy under both criteria has the trunk reservation
form known for loss systems: the rare, valuable type is accepted whenever a
server is idle, and the common one only while fewer than a threshold of servers
are busy.

Sub-tests:
    Case 1: Discounted reward: value iteration, policy iteration and modified
            policy iteration, each with standard, Gauss-Seidel and SOR updates,
            give the exact optimal policy, and values within the tolerance
            handed to solve() of the exact ones.
    Case 2: Average reward: the three algorithms, with standard updates (the
            only ones available under this criterion), give the exact
            gain-optimal policy, and values whose differences v(s) - v(0) are
            the bias. Every action leads to the empty system with positive
            probability, so every policy is unichain and aperiodic, which value
            iteration needs under this criterion (Puterman, 1994, Section 8.5).
    Case 3: The model given as CSV files, as the former test2.py gave it,
            gives the same policies and values as the lists, under each
            algorithm.

The tolerances keep clear of what double precision allows on values of this
size (about 1e5): at a tolerance of 1e-9 the stopping threshold of the
discounted criterion, epsilon(1 - lambda)/lambda, would be smaller than the
rounding of the values, and the solver would run to its iteration limit. The
average-reward iterates grow by the gain in every iteration and reach about
1e6, so the same holds there below about 1e-9. Nothing may be printed by any
of the runs.

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.

References:
    Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic
    Dynamic Programming. Wiley. https://doi.org/10.1002/9780470316887
'''

import os
import sys
import tempfile

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import checkoptimality, elementwisetranmat, exactaverage, exactdiscounted


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

# The queueing problem, as in the former test2.py
CAPACITY = 100
ARRIVALRATE1 = 50.0
ARRIVALRATE2 = 10.0
SERVICERATE = 3.0 / 4.0
REWARD1 = 1.0
REWARD2 = 1000.0
DISCOUNT = 0.99
TEST2MEAN = 17075.0          # the rounded mean of the discounted values in the former test2.py

TOLERANCE = 1e-6             # the epsilon handed to solve() under discounting
AVERAGETOLERANCE = 1e-8      # and under the average-reward criterion

# The span stopping rule of the average-reward criterion bounds the error of
# the gain rather than of the bias, and this problem mixes slowly: the bias
# comes out within about 70 times the tolerance (7e-7 here), so it is compared
# at a tolerance set by measurement.
BIASTOL = 1e-5

# The optimal action of every state with more than one action must beat the
# runner-up by at least MINGAP, far more than the tolerance (the smallest gap
# is 0.045 here)
MINGAP = 1e-3

ALGORITHMS = (("value iteration", "vi"), ("policy iteration", "pi"), ("modified policy iteration", "mpi"))
DISCOUNTEDUPDATES = (("standard updates", dict(update="standard")),
                     ("Gauss-Seidel updates", dict(update="gs")),
                     ("SOR updates with relaxation 1.2", dict(update="sor", SORrelaxation=1.2)))

# The discount factor in mdp() is not used under the average-reward criterion,
# but has to be valid
AVERAGEDISCOUNT = 0.5


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def exitprobabilities(busy):
    '''The probabilities that the next arrival finds j = 0, ..., busy servers busy, with busy busy now.'''
    rate = ARRIVALRATE1 + ARRIVALRATE2
    probabilities = np.zeros(busy + 1)
    finishing = 1.0   # the probability that the servers above j all finish first
    for j in range(busy, -1, -1):
        probabilities[j] = finishing * rate / (j * SERVICERATE + rate)
        finishing *= j * SERVICERATE / (j * SERVICERATE + rate)
    return probabilities


def queueproblem():
    '''The rewards and transitions of the queueing problem: action 0 rejects, action 1 accepts.'''
    share1 = ARRIVALRATE1 / (ARRIVALRATE1 + ARRIVALRATE2)
    share2 = ARRIVALRATE2 / (ARRIVALRATE1 + ARRIVALRATE2)
    rewards, transitions = [], []
    for reward in (REWARD1, REWARD2):
        for busy in range(CAPACITY + 1):
            staterewards, statetransitions = [], []
            for accept in ((0, 1) if busy < CAPACITY else (0,)):
                probabilities = exitprobabilities(busy + accept)
                row = {}
                for j, p in enumerate(probabilities):
                    row[j] = p * share1
                for j, p in enumerate(probabilities):
                    row[CAPACITY + 1 + j] = p * share2
                staterewards.append(reward if accept else 0.0)
                statetransitions.append(row)
            rewards.append(staterewards)
            transitions.append(statetransitions)
    return rewards, transitions


def solve(algorithm, setting, criterion="discounted", files=None):
    '''One algorithm under one setting, the model given as lists or files; returns the policy, the values and anything printed.'''
    def run():
        mdl = mdpsolver.model()
        if files is None:
            mdl.mdp(discount=discount, rewards=REWARDS, tranMatElementwise=ELEMENTWISE)
        else:
            mdl.mdp(discount=discount, rewardsFromFile=files[0], tranMatFromFile=files[1])
        mdl.solve(algorithm=algorithm, tolerance=tolerance, criterion=criterion, **setting)
        return list(mdl.getPolicy()), np.array(mdl.getValueVector(), dtype=float)

    discount, tolerance = (DISCOUNT, TOLERANCE) if criterion == "discounted" else (AVERAGEDISCOUNT, AVERAGETOLERANCE)
    (policy, values), output = quietly(run)
    return policy, values, output


def trunkreservation(policy):
    '''The threshold below which the common type is accepted, if the policy has the trunk reservation form, else None.'''
    common = policy[:CAPACITY]
    rare = policy[CAPACITY + 1:2 * CAPACITY + 1]
    threshold = sum(common)
    if common != [1] * threshold + [0] * (CAPACITY - threshold) or rare != [1] * CAPACITY:
        return None
    return threshold


# ------------------------------------------------------------------
# THE MODEL AND ITS EXACT SOLUTIONS
# ------------------------------------------------------------------

REWARDS, TRANSITIONS = queueproblem()
ELEMENTWISE = elementwisetranmat(TRANSITIONS)
NSTATES = len(REWARDS)

# The exact solutions are the whole test, so they are checked against the
# optimality equations, and the model for the features the cases rely on,
# before they are used.
for s, state in enumerate(TRANSITIONS):
    for a, probs in enumerate(state):
        if abs(sum(probs.values()) - 1.0) > 1e-12:
            sys.exit("Validation test failed at initialization: the probabilities of action %d in state %d sum to %.15f, not 1." % (a, s, sum(probs.values())))
        if not probs.get(0, 0.0) > 0.0:
            sys.exit("Validation test failed at initialization: action %d in state %d cannot lead to the empty system, so the policies need not be unichain and aperiodic." % (a, s))

exactpolicy, exactvalues = exactdiscounted(REWARDS, TRANSITIONS, DISCOUNT)
checkoptimality("queueing problem at discount %g" % DISCOUNT, REWARDS, TRANSITIONS,
                exactpolicy, exactvalues, DISCOUNT, MINGAP)
averagepolicy, averagegain, averagebias = exactaverage(REWARDS, TRANSITIONS)
checkoptimality("queueing problem under the average-reward criterion", REWARDS, TRANSITIONS,
                averagepolicy, averagebias, 1.0, MINGAP, averagegain)

if round(float(np.mean(exactvalues))) != TEST2MEAN:
    sys.exit("Validation test failed at initialization: the mean of the exact discounted values is %.4f, which does not round to the %.0f of the former test2.py, so the model is not the one it used." % (np.mean(exactvalues), TEST2MEAN))

for name, policy in (("discounted", exactpolicy), ("average-reward", averagepolicy)):
    threshold = trunkreservation(policy)
    if threshold is None or not 0 < threshold < CAPACITY:
        sys.exit("Validation test failed at initialization: the exact %s policy is not a trunk reservation with a threshold inside the capacity." % name)


# ------------------------------------------------------------------
# CASE 1: Discounted reward
# ------------------------------------------------------------------

discountedresults = {}
for algorithmlabel, algorithm in ALGORITHMS:
    for updatelabel, setting in DISCOUNTEDUPDATES:
        label = "%s with %s" % (algorithmlabel, updatelabel)
        policy, values, output = solve(algorithm, setting)

        if output:
            sys.exit("Validation test failed at case 1: %s printed %r." % (label, output.strip()))
        if policy != exactpolicy:
            sys.exit("Validation test failed at case 1: %s gives a policy that differs from the exact one in %d states." % (label, sum(a != b for a, b in zip(policy, exactpolicy)) if len(policy) == NSTATES else NSTATES))
        if values.shape != exactvalues.shape or not np.all(np.isfinite(values)):
            sys.exit("Validation test failed at case 1: %s gives %d values, not all finite, for %d states." % (label, values.size, NSTATES))
        difference = float(np.max(np.abs(values - exactvalues)))
        if difference > TOLERANCE:
            sys.exit("Validation test failed at case 1: %s gives values that differ by %.3e from the exact ones, which is more than %.1e." % (label, difference, TOLERANCE))
        discountedresults[algorithm, updatelabel] = (policy, values)


# ------------------------------------------------------------------
# CASE 2: Average reward
# ------------------------------------------------------------------

averageresults = {}
for algorithmlabel, algorithm in ALGORITHMS:
    label = "%s under the average-reward criterion" % algorithmlabel
    policy, values, output = solve(algorithm, dict(update="standard"), criterion="average")

    if output:
        sys.exit("Validation test failed at case 2: %s printed %r." % (label, output.strip()))
    if policy != averagepolicy:
        sys.exit("Validation test failed at case 2: %s gives a policy that differs from the exact one in %d states." % (label, sum(a != b for a, b in zip(policy, averagepolicy)) if len(policy) == NSTATES else NSTATES))
    if values.shape != averagebias.shape or not np.all(np.isfinite(values)):
        sys.exit("Validation test failed at case 2: %s gives %d values, not all finite, for %d states." % (label, values.size, NSTATES))
    difference = float(np.max(np.abs((values - values[0]) - averagebias)))
    if difference > BIASTOL:
        sys.exit("Validation test failed at case 2: %s gives values whose differences from the first differ by %.3e from the bias, which is more than %.1e." % (label, difference, BIASTOL))
    averageresults[algorithm] = (policy, values)


# ------------------------------------------------------------------
# CASE 3: The model given as files
# ------------------------------------------------------------------

# The files hold the same numbers as the lists (repr() round-trips a float) in
# the same order, so the results must be the same exactly.
with tempfile.TemporaryDirectory() as directory:
    rewardfile = os.path.join(directory, "rewards.csv")
    tranmatfile = os.path.join(directory, "transitions.csv")
    with open(rewardfile, "w", newline="") as file:
        file.write("stateFrom,action,reward\n")
        for s, state in enumerate(REWARDS):
            for a, reward in enumerate(state):
                file.write("%d,%d,%r\n" % (s, a, reward))
    with open(tranmatfile, "w", newline="") as file:
        file.write("stateFrom,action,stateTo,probability\n")
        for s, a, j, p in ELEMENTWISE:
            file.write("%d,%d,%d,%r\n" % (s, a, j, float(p)))

    for algorithmlabel, algorithm in ALGORITHMS:
        for criterion, reference in (("discounted", discountedresults[algorithm, "standard updates"]),
                                     ("average", averageresults[algorithm])):
            label = "%s under the %s criterion, with the model given as files," % (algorithmlabel, criterion)
            policy, values, output = solve(algorithm, dict(update="standard"), criterion, files=(rewardfile, tranmatfile))

            if output:
                sys.exit("Validation test failed at case 3: %s printed %r." % (label, output.strip()))
            if policy != reference[0] or values.shape != reference[1].shape or not np.array_equal(values, reference[1]):
                sys.exit("Validation test failed at case 3: %s does not give the policy and values it gives with the model given as lists." % label)


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
