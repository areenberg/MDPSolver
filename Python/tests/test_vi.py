'''
UNIT TEST FOR VALUE ITERATION (algorithm="vi" in model.solve()).

Value iteration is run on two problems whose solutions are known in closed form,
the hub-and-spoke problem and the secretary problem (described in
knownsolutions.py), under every setting of solve() that applies to it, and its
policy and value vector are compared with the known solutions. Both closed
forms are checked against the optimality equations before they are used, and so
are the margins by which the optimal actions win, so that a policy comparison
cannot hinge on rounding.

Sub-tests:
    Case 1: Infeasible arguments abort with an error message: an unknown
            algorithm, update method or optimality criterion, a tolerance
            outside (0, 1), an SOR relaxation outside (0, 2), an initial value
            vector or policy of the wrong length, an initial policy with an
            action that its state does not have, and Gauss-Seidel or SOR
            updates under the average-reward criterion (see below).
    Case 2: Arguments that value iteration does not use are accepted silently
            and change nothing: parallel=True with Gauss-Seidel or SOR updates
            (parallel computing is only available with standard updates), an
            SOR relaxation with other updates, a partial evaluation limit, and
            an initial policy.
    Case 3: Discounted reward: under every setting, the policy of both problems
            is the known optimal one, and every value lies within the tolerance
            handed to solve() of the known one, since the solution is
            epsilon-optimal for that tolerance. All settings therefore give the
            same policy and the same values (to twice the tolerance). Without
            post-processing, standard updates give values that are off by a
            constant only.
    Case 4: Discounted reward: the values increase with the discount factor,
            under every setting. With non-negative rewards that do not depend
            on lambda, the value of every policy is a power series in lambda
            with non-negative coefficients, and the optimal value is the
            largest of them, so no value can fall as lambda grows. In the
            hub-and-spoke problem every value rises (which case 3 implies
            already, as the known values rise by far more than the tolerance).
            The check that does not rest on case 3 is on the plain discounted
            secretary problem, with the reward i/n for selecting candidate i
            whatever lambda, whose solution the test does not compute: no value
            may fall by more than twice the tolerance from one discount factor
            to the next. Values may stay flat there, where selecting at once is
            best (at small lambda even the first candidate is selected, worth
            1/n), but the start must rise from the smallest discount factor to
            the largest, where waiting pays.
    Case 5: Average reward: the policy of the hub-and-spoke problem is the
            gain-optimal one, and the values reflect the bias. MDPSolver returns
            the iterates of value iteration, which grow by the gain in every
            iteration, so their level depends on the number of iterations; the
            differences v(s) - v(0) are compared with the bias instead. Value
            iteration converges under this criterion when every policy is
            unichain and aperiodic (Puterman, 1994, Section 8.5), which is
            checked for every policy of the problem. (The secretary problem is
            left out, as every policy has gain 0 in its absorbing state.)

Gauss-Seidel and SOR updates must be rejected under the average-reward
criterion, which they do not solve: their in-place updates make the values grow
by g/(1 - pi L 1) per sweep rather than by the gain g (L being the part of the
transition matrix that leads to lower-numbered states and pi the stationary
distribution), so their supremum-norm stopping rule is never met, and the
policy they end with need not be gain-optimal. No stopping rule mends that, as
the greedy step itself then maximizes another criterion.

Nothing may be printed by any of the runs (the final check of the value vector
prints a message if it finds the values outside their possible range).

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.

References:
    Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic
    Dynamic Programming. Wiley. https://doi.org/10.1002/9780470316887
'''

import sys

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import (WINPROBABILITY, checkoptimality, densetranmat, hubaverage,
                            hubdiscounted, hubproblem, secretaryproblem, secretarysolution,
                            unichainaperiodic)


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

ALGORITHM = "vi"
TOLERANCE = 1e-9   # the epsilon handed to solve()

# The span stopping rule of the average-reward criterion bounds the error of
# the gain rather than of the bias, so the bias is compared at a tolerance set
# by measurement: the differences lie within 2e-10 of it here.
BIASTOL = 1e-8

# The optimal action of every state with more than one action must beat the
# runner-up by at least MINGAP, far more than the tolerance (the smallest gap
# is 1.7e-3 here).
MINGAP = 1e-4

# Discount factors for the hub-and-spoke problem, in increasing order, from
# where resting is best, past where the short venture is, to where the long one
# is and the poor one is quit
HUBDISCOUNTS = (0.3, 0.6, 0.8, 0.9, 0.95, 0.99)

# The discount factor in mdp() is not used under the average-reward criterion,
# but has to be valid
AVERAGEDISCOUNT = 0.5

# Discount factors for the secretary problem, and, in increasing order, for the
# plain discounted one of case 4
SECRETARYDISCOUNTS = (0.5, 0.9, 0.999)
PLAINDISCOUNTS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)

# An initial value vector far above every solution, for the warm starts
FARVALUE = 1e4

# The settings of solve() under which value iteration is run. Parallel
# computing is the default and is only available with standard updates. The
# initial vectors are given by name and made to fit each problem (see
# arguments() below). SOR relaxations above 1.5 are left out: in the secretary
# problem at discount 0.999 the rounding errors of SOR updates, amplified by
# about relaxation*lambda/(2 - relaxation) per step back along the chain of
# candidates, keep the change per sweep above the stopping threshold
# epsilon(1 - lambda)/(2 lambda) = 5e-13, so from 1.6 the solver runs to its
# iteration limit (10^6 sweeps) without saying so, though with the right values.
DISCOUNTEDSETTINGS = (
    ("standard updates, in parallel", dict(update="standard")),
    ("standard updates, serially", dict(update="standard", parallel=False)),
    ("Gauss-Seidel updates", dict(update="gs")),
    ("SOR updates with relaxation 0.8", dict(update="sor", SORrelaxation=0.8)),
    ("SOR updates with relaxation 1.2", dict(update="sor", SORrelaxation=1.2)),
    ("SOR updates with relaxation 1.5", dict(update="sor", SORrelaxation=1.5)),
    ("standard updates from a zero initial value vector", dict(update="standard", initValueVector="zeros")),
    ("Gauss-Seidel updates from an initial value vector far above the solution", dict(update="gs", initValueVector="far")),
    ("SOR updates from an initial value vector far above the solution", dict(update="sor", SORrelaxation=1.2, initValueVector="far")),
    ("standard updates without the final check", dict(update="standard", makeFinalCheck=False)),
    ("Gauss-Seidel updates without post-processing", dict(update="gs", postProcessing=False)),
)

AVERAGESETTINGS = (
    ("standard updates, in parallel", dict(update="standard")),
    ("standard updates, serially", dict(update="standard", parallel=False)),
    ("standard updates from a zero initial value vector", dict(update="standard", initValueVector="zeros")),
    ("standard updates from an initial value vector far above the iterates", dict(update="standard", initValueVector="far")),
    ("standard updates without the final check", dict(update="standard", makeFinalCheck=False)),
    ("standard updates without post-processing", dict(update="standard", postProcessing=False)),
)

# Case 1: arguments that must abort, with a word the error message must contain
INFEASIBLE = (
    ("an unknown algorithm", dict(algorithm="value iteration"), "algorithm"),
    ("an unknown update method", dict(update="jacobi"), "update"),
    ("an unknown optimality criterion", dict(criterion="total"), "criterion"),
    ("a tolerance of zero", dict(tolerance=0.0), "tolerance"),
    ("a negative tolerance", dict(tolerance=-1e-6), "tolerance"),
    ("a tolerance of one", dict(tolerance=1.0), "tolerance"),
    ("a tolerance above one", dict(tolerance=2.5), "tolerance"),
    ("an SOR relaxation of zero", dict(update="sor", SORrelaxation=0.0), "relaxation"),
    ("a negative SOR relaxation", dict(update="sor", SORrelaxation=-0.5), "relaxation"),
    ("an SOR relaxation of two", dict(update="sor", SORrelaxation=2.0), "relaxation"),
    ("an initial value vector one too short", dict(initValueVector="short"), "initial value vector"),
    ("an initial value vector one too long", dict(initValueVector="long"), "initial value vector"),
    ("an initial policy one too short", dict(initPolicy="short"), "initial policy"),
    ("an initial policy one too long", dict(initPolicy="long"), "initial policy"),
    ("an initial policy with an action beyond the last of its state", dict(initPolicy="beyond"), "initial policy"),
    ("an initial policy with a negative action", dict(initPolicy="negative"), "initial policy"),
    ("Gauss-Seidel updates under the average-reward criterion", dict(criterion="average", update="gs"), "average"),
    ("SOR updates under the average-reward criterion", dict(criterion="average", update="sor", SORrelaxation=1.2), "average"),
)

# Case 2: arguments that value iteration does not use, each with the setting it
# must reproduce exactly (the same code runs on the same numbers)
IGNORED = (
    ("parallel=True with Gauss-Seidel updates", dict(update="gs", parallel=True), dict(update="gs", parallel=False)),
    ("parallel=True with SOR updates", dict(update="sor", SORrelaxation=1.2, parallel=True), dict(update="sor", SORrelaxation=1.2, parallel=False)),
    ("an SOR relaxation of 5 with standard updates", dict(update="standard", SORrelaxation=5.0), dict(update="standard")),
    ("an SOR relaxation of 5 with Gauss-Seidel updates", dict(update="gs", SORrelaxation=5.0), dict(update="gs")),
    ("a partial evaluation limit of 0", dict(update="standard", parIterLim=0), dict(update="standard")),
    ("an initial policy", dict(update="standard", initPolicy="last"), dict(update="standard")),
)
IGNOREDDISCOUNT = 0.9


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def arguments(setting, rewards):
    '''The arguments of solve() for a setting, with the initial vectors given by name made to fit the problem.'''
    result = dict(setting)
    n = len(rewards)
    values = {"zeros": [0.0] * n, "far": [FARVALUE] * n,
              "short": [0.0] * (n - 1), "long": [0.0] * (n + 1)}
    policies = {"zeros": [0] * n, "last": [len(r) - 1 for r in rewards],
                "short": [0] * (n - 1), "long": [0] * (n + 1),
                "beyond": [0] * (n - 1) + [len(rewards[-1])], "negative": [-1] + [0] * (n - 1)}
    if isinstance(result.get("initValueVector"), str):
        result["initValueVector"] = values[result["initValueVector"]]
    if isinstance(result.get("initPolicy"), str):
        result["initPolicy"] = policies[result["initPolicy"]]
    return result


def solve(rewards, transitions, discount, setting, criterion="discounted"):
    '''The algorithm under one setting; returns the policy, the value vector and anything printed.'''
    def run():
        mdl = mdpsolver.model()
        mdl.mdp(discount=discount, rewards=rewards, tranMatWithZeros=densetranmat(transitions))
        mdl.solve(algorithm=ALGORITHM, tolerance=TOLERANCE, criterion=criterion, **arguments(setting, rewards))
        return list(mdl.getPolicy()), np.array(mdl.getValueVector(), dtype=float)

    (policy, values), output = quietly(run)
    return policy, values, output


def checkrun(case, label, result, exactpolicy, exactvalues=None, allowed=None):
    '''Ends the test if a run printed anything, or its policy or values differ from the known ones.'''
    policy, values, output = result
    if output:
        sys.exit("Validation test failed at case %d: %s printed %r." % (case, label, output.strip()))
    if policy != exactpolicy:
        sys.exit("Validation test failed at case %d: %s gives the policy %s, where the known one is %s." % (case, label, policy, exactpolicy))
    if exactvalues is None:
        return
    if values.shape != exactvalues.shape or not np.all(np.isfinite(values)):
        sys.exit("Validation test failed at case %d: %s gives the values %s." % (case, label, values))
    difference = float(np.max(np.abs(values - exactvalues)))
    if difference > allowed:
        sys.exit("Validation test failed at case %d: %s gives values that differ by %.3e from the known ones, which is more than %.1e." % (case, label, difference, allowed))


# ------------------------------------------------------------------
# THE KNOWN SOLUTIONS
# ------------------------------------------------------------------

# The known solutions are the whole test, so they are checked against the
# optimality equations, and the problems for the features the cases rely on,
# before they are used.
HUBREWARDS, HUBTRANSITIONS = hubproblem()

hubsolutions = {}
for discount in HUBDISCOUNTS:
    hubsolutions[discount] = hubdiscounted(discount)
    checkoptimality("hub-and-spoke problem at discount %g" % discount, HUBREWARDS, HUBTRANSITIONS,
                    *hubsolutions[discount], discount, MINGAP)

averagepolicy, averagegain, averagebias = hubaverage()
checkoptimality("hub-and-spoke problem under the average-reward criterion", HUBREWARDS, HUBTRANSITIONS,
                averagepolicy, averagebias, 1.0, MINGAP, averagegain)

# the policy has to change with the discount factor, in the hub and in a venture
hubchoices = {hubsolutions[discount][0][0] for discount in HUBDISCOUNTS}
venturechoices = {tuple(hubsolutions[discount][0][1:]) for discount in HUBDISCOUNTS}
if len(hubchoices) < 3 or len(venturechoices) < 2:
    sys.exit("Validation test failed at initialization: the optimal policy of the hub-and-spoke problem does not change enough with the discount factor (hub choices %s), so the policy is not tested on the discount." % sorted(hubchoices))

if any(reward < 0.0 for state in HUBREWARDS for reward in state):
    sys.exit("Validation test failed at initialization: the hub-and-spoke problem has a negative reward, so its values need not increase with the discount factor.")

if list(HUBDISCOUNTS) != sorted(HUBDISCOUNTS) or list(PLAINDISCOUNTS) != sorted(PLAINDISCOUNTS):
    sys.exit("Validation test failed at initialization: the discount factors of case 4 are not in increasing order.")

if any(reward < 0.0 for state in secretaryproblem(0.5, compensated=False)[0] for reward in state):
    sys.exit("Validation test failed at initialization: the plain discounted secretary problem has a negative reward, so its values need not increase with the discount factor.")

for low, high in zip(HUBDISCOUNTS[:-1], HUBDISCOUNTS[1:]):
    if np.min(hubsolutions[high][1] - hubsolutions[low][1]) <= 2.0 * TOLERANCE:
        sys.exit("Validation test failed at initialization: the known values of the hub-and-spoke problem do not increase by more than twice the tolerance from discount %g to %g, so case 4 could not tell." % (low, high))

nonconvergent = unichainaperiodic(HUBREWARDS, HUBTRANSITIONS)
if nonconvergent is not None:
    sys.exit("Validation test failed at initialization: the policy %s of the hub-and-spoke problem is not unichain and aperiodic, so value iteration need not converge under the average-reward criterion." % (list(nonconvergent),))

secretarysolutions = {}
for discount in SECRETARYDISCOUNTS:
    cutoff, policy, values = secretarysolution(discount)
    rewards, transitions = secretaryproblem(discount)
    checkoptimality("secretary problem at discount %g" % discount, rewards, transitions, policy, values, discount, MINGAP)
    secretarysolutions[discount] = (rewards, transitions, policy, values)

    if abs(values[0] - WINPROBABILITY) > 5e-7:
        sys.exit("Validation test failed at initialization: the start of the secretary problem is worth %.6f at discount %g, not the classical %.6f." % (values[0], discount, WINPROBABILITY))

if FARVALUE <= max(np.max(np.abs(solution[1])) for solution in hubsolutions.values()) or \
        FARVALUE <= max(np.max(np.abs(solution[3])) for solution in secretarysolutions.values()):
    sys.exit("Validation test failed at initialization: the far initial value vector is not above the solutions.")


# ------------------------------------------------------------------
# CASE 1: Infeasible arguments
# ------------------------------------------------------------------

for label, setting, word in INFEASIBLE:
    given = dict(algorithm=ALGORITHM, tolerance=TOLERANCE)
    given.update(arguments(setting, HUBREWARDS))

    mdl = mdpsolver.model()
    mdl.mdp(discount=0.9, rewards=HUBREWARDS, tranMatWithZeros=densetranmat(HUBTRANSITIONS))
    try:
        mdl.solve(**given)
    except SystemExit as error:
        message = str(error.code)
    else:
        sys.exit("Validation test failed at case 1: %s was accepted." % label)

    if not message.startswith("Error:") or word not in message.lower():
        sys.exit("Validation test failed at case 1: %s aborted with the message %r, which is not an error message about the %s." % (label, message, word))


# ------------------------------------------------------------------
# CASE 2: Arguments that value iteration does not use
# ------------------------------------------------------------------

for label, setting, reference in IGNORED:
    try:
        result = solve(HUBREWARDS, HUBTRANSITIONS, IGNOREDDISCOUNT, setting)
    except SystemExit as error:
        sys.exit("Validation test failed at case 2: %s aborted with the message %r." % (label, str(error.code)))
    expected = solve(HUBREWARDS, HUBTRANSITIONS, IGNOREDDISCOUNT, reference)

    if result[2]:
        sys.exit("Validation test failed at case 2: %s printed %r." % (label, result[2].strip()))
    if result[0] != expected[0]:
        sys.exit("Validation test failed at case 2: %s changes the policy to %s, where it should be %s." % (label, result[0], expected[0]))
    if result[1].shape != expected[1].shape or not np.array_equal(result[1], expected[1]):
        difference = float(np.max(np.abs(result[1] - expected[1]))) if result[1].shape == expected[1].shape else np.inf
        sys.exit("Validation test failed at case 2: %s changes the values, by up to %.3e, where they should be the same." % (label, difference))


# ------------------------------------------------------------------
# CASE 3: Discounted reward against the known solutions
# ------------------------------------------------------------------

# Under every setting, both problems at every discount factor. The values have
# to lie within the tolerance of the known ones (they lie within half of it
# here, as the stopping rules promise).
hubvalues = {}
for settinglabel, setting in DISCOUNTEDSETTINGS:
    for discount in HUBDISCOUNTS:
        policy, values = hubsolutions[discount]
        result = solve(HUBREWARDS, HUBTRANSITIONS, discount, setting)
        checkrun(3, "the hub-and-spoke problem at discount %g with %s" % (discount, settinglabel), result,
                 policy, values, TOLERANCE)
        hubvalues[settinglabel, discount] = result[1]

    for discount in SECRETARYDISCOUNTS:
        rewards, transitions, policy, values = secretarysolutions[discount]
        result = solve(rewards, transitions, discount, setting)
        checkrun(3, "the secretary problem at discount %g with %s" % (discount, settinglabel), result,
                 policy, values, TOLERANCE)

# Post-processing adds the same constant to every value (Puterman, 1994,
# Eq. 6.6.12), so standard updates without it give the known policy and values
# that are off by a constant: their differences from the known values may
# spread by twice the tolerance at most.
for discount in HUBDISCOUNTS:
    policy, values = hubsolutions[discount]
    result = solve(HUBREWARDS, HUBTRANSITIONS, discount, dict(update="standard", postProcessing=False))
    label = "the hub-and-spoke problem at discount %g with standard updates without post-processing" % discount
    checkrun(3, label, result, policy)
    spread = float(np.ptp(result[1] - values))
    if not np.all(np.isfinite(result[1])) or spread > 2.0 * TOLERANCE:
        sys.exit("Validation test failed at case 3: %s gives values whose differences from the known ones spread by %.3e, so they are off by more than a constant." % (label, spread))


# ------------------------------------------------------------------
# CASE 4: The values increase with the discount factor
# ------------------------------------------------------------------

for settinglabel, setting in DISCOUNTEDSETTINGS:
    for low, high in zip(HUBDISCOUNTS[:-1], HUBDISCOUNTS[1:]):
        increase = hubvalues[settinglabel, high] - hubvalues[settinglabel, low]
        if not np.all(increase > 0.0):
            state = int(np.argmin(increase))
            sys.exit("Validation test failed at case 4: with %s, the value of state %d of the hub-and-spoke problem does not rise from %.12g at discount %g to %.12g at discount %g." % (settinglabel, state, hubvalues[settinglabel, low][state], low, hubvalues[settinglabel, high][state], high))

    plainvalues = []
    for discount in PLAINDISCOUNTS:
        rewards, transitions = secretaryproblem(discount, compensated=False)
        result = solve(rewards, transitions, discount, setting)
        label = "the plain discounted secretary problem at discount %g with %s" % (discount, settinglabel)
        if result[2]:
            sys.exit("Validation test failed at case 4: %s printed %r." % (label, result[2].strip()))
        if result[1].shape != (len(rewards),) or not np.all(np.isfinite(result[1])):
            sys.exit("Validation test failed at case 4: %s gives the values %s." % (label, result[1]))
        plainvalues.append(result[1])

    for (low, lowvalues), (high, highvalues) in zip(zip(PLAINDISCOUNTS, plainvalues), zip(PLAINDISCOUNTS[1:], plainvalues[1:])):
        fall = float(np.max(lowvalues - highvalues))
        if fall > 2.0 * TOLERANCE:
            state = int(np.argmax(lowvalues - highvalues))
            sys.exit("Validation test failed at case 4: with %s, the value of state %d of the plain discounted secretary problem falls from %.12g at discount %g to %.12g at discount %g." % (settinglabel, state, lowvalues[state], low, highvalues[state], high))

    if plainvalues[-1][0] - plainvalues[0][0] <= 2.0 * TOLERANCE:
        sys.exit("Validation test failed at case 4: with %s, the value of the start of the plain discounted secretary problem does not rise from %.12g at discount %g to %.12g at discount %g." % (settinglabel, plainvalues[0][0], PLAINDISCOUNTS[0], plainvalues[-1][0], PLAINDISCOUNTS[-1]))


# ------------------------------------------------------------------
# CASE 5: Average reward
# ------------------------------------------------------------------

for settinglabel, setting in AVERAGESETTINGS:
    label = "the hub-and-spoke problem under the average-reward criterion with %s" % settinglabel
    result = solve(HUBREWARDS, HUBTRANSITIONS, AVERAGEDISCOUNT, setting, criterion="average")
    checkrun(5, label, result, averagepolicy)

    relative = result[1] - result[1][0]
    difference = float(np.max(np.abs(relative - averagebias))) if relative.shape == averagebias.shape else np.inf
    if not np.all(np.isfinite(result[1])) or difference > BIASTOL:
        sys.exit("Validation test failed at case 5: %s gives the values %s, whose differences from the first differ by %.3e from the bias %s, which is more than %.1e." % (label, result[1], difference, averagebias, BIASTOL))


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
