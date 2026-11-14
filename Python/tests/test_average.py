'''
UNIT TEST FOR THE GAIN AND THE BIAS UNDER THE AVERAGE REWARD CRITERION
(model.getGain() and model.getBias()).

Under the average reward criterion, solve() ends every algorithm with an
improvement step v' = Lv. The smallest and largest change in that step, the
smallest and largest entry of v' - v, bound the gain: the policy found chose
its actions in that step, so its gain is an average of v' - v (weighted by its
limiting distribution), and the optimal gain is at least the policy's and at
most the largest entry (Puterman, 1994, Section 8.5). The bounds hold for any
v, so also when solve() stops at a limit. getGain() returns their midpoint, and
getGain(bound="lower") and getGain(bound="upper") the bounds themselves.

The values grow by about the gain in every iteration, so only their
differences are meaningful: they approach the differences between the biases
of the states. getBias(s, referenceIndex=k) returns the value of state s
minus the value of state k, the bias of state s with the bias of state k set
to 0.

Sub-tests:
    Case 1: The hub-and-spoke problem of knownsolutions.py, whose gain is known
            in closed form, under value iteration, policy iteration and
            modified policy iteration, with serial and parallel computation, and
            at a loose and a tight tolerance: the bounds contain the gain and
            are less than the tolerance apart, getGain() is their midpoint and
            lies within half the tolerance of the gain, and the run converged.
    Case 2: Random unichain models, at a loose tolerance: the bounds contain
            both the gain of the policy found and the optimal gain, each
            computed exactly by solving linear equations.
    Case 3: A periodic model, a cycle of three states with one action each,
            where value iteration and the evaluation of policy iteration never
            converge: solve() stops at its limit with a ConvergenceWarning, and
            the bounds still contain the gain, while their midpoint does not
            equal it.
    Case 4: The hub-and-spoke problem and the random models of case 2, whose
            biases are known (in closed form, and by solving linear
            equations), at a tight tolerance, under every algorithm: for
            every state and every reference state, the bias is a float, the
            value of the state minus that of the reference state (so exactly
            0 for the reference state itself), and lies within BIASTOL of the
            known bias with that of the reference state subtracted. The
            default reference state is state 0, and NumPy integers work as
            well.
    Case 5: getGain() and getBias() before solve() raise a RuntimeError, and
            so do they after a run under the discounted criterion, also when
            an earlier run on the same object was under the average criterion.
            A run under the average criterion after a discounted one gives the
            gain and the bias. The bound is selected in any case ("Lower",
            "UPPER"), and any other bound raises a ValueError. A state or a
            reference state outside the model raises an IndexError, and one
            that is not an integer a TypeError.

References:
    Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic
    Dynamic Programming. Wiley. https://doi.org/10.1002/9780470316887

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import sys
import warnings

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import arrayinput, exactaverage, hubaverage, hubproblem, matrices


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

DISCOUNT = 0.9             # mdp() needs one; the average reward criterion ignores it
TOLERANCES = (1e-3, 1e-9)  # a loose and a tight epsilon handed to solve()
LOOSE = TOLERANCES[0]
TIGHT = TOLERANCES[1]
ROUNDING = 1e-12           # the margin for rounding when the bounds are compared with an exact gain
BIASTOL = 1e-8             # how far a bias may lie from the known one at the tight tolerance (the errors are about 1e-10)

ALGORITHMS = (("value iteration", "vi"), ("policy iteration", "pi"), ("modified policy iteration", "mpi"))

HUBREWARDS, HUBTRANSITIONS = hubproblem()
HUBGAIN, HUBBIAS = hubaverage()[1:]

# Case 2: random models in which every action leads to state 0 with a
# probability of at least MINPROB, so every policy is unichain (state 0 is in
# its only recurrent class) and aperiodic (state 0 can stay in state 0)
NSTATES, NACTIONS, NNEXT = 40, 3, 4
MINPROB = 0.05
SEEDS = range(5)

# Case 3: a cycle 0 -> 1 -> 2 -> 0 with the reward CYCLEREWARD in state 2 only,
# whose gain is CYCLEREWARD / 3, while the changes of a step are 0 and CYCLEREWARD
CYCLEREWARD = 3.0
CYCLEREWARDS = [[0.0], [0.0], [CYCLEREWARD]]
CYCLETRANSITIONS = [[{1: 1.0}], [{2: 1.0}], [{0: 1.0}]]
CYCLEGAIN = CYCLEREWARD / 3.0

if not HUBGAIN > 0.0:
    sys.exit("Validation test failed at initialization: the gain of the hub-and-spoke problem is %r, where it should be positive." % HUBGAIN)
# the biases of the hub-and-spoke problem differ between the states by far
# more than BIASTOL, so a wrong reference state cannot pass case 4
if HUBBIAS[0] != 0.0 or np.min(np.abs(np.subtract.outer(HUBBIAS, HUBBIAS))[~np.eye(len(HUBBIAS), dtype=bool)]) < 100 * BIASTOL:
    sys.exit("Validation test failed at initialization: the biases of the hub-and-spoke problem, %s, are not 0 in the hub and far apart." % HUBBIAS)
if np.isclose(CYCLEGAIN, CYCLEREWARD / 2.0):
    sys.exit("Validation test failed at initialization: the gain of the cycle equals the midpoint of its bounds, so case 3 could not tell them apart.")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def solve(rewards, transitions, **settings):
    '''A new model object, solved under the average reward criterion; anything solve() prints ends the test.'''
    def run():
        mdl = mdpsolver.model()
        mdl.mdp(discount=DISCOUNT, **arrayinput(rewards, transitions, bycolumn=True))
        mdl.solve(criterion="average", **settings)
        return mdl

    mdl, output = quietly(run)
    if output:
        sys.exit("Validation test failed: solve() with %s printed %r." % (settings, output.strip()))
    return mdl


def bounds(mdl):
    '''The lower bound, the midpoint and the upper bound that getGain() returns.'''
    return mdl.getGain(bound="lower"), mdl.getGain(), mdl.getGain(bound="upper")


def checkbounds(case, label, lower, upper, gain, what="the gain"):
    '''Ends the test unless the bounds contain the gain.'''
    if not (np.isfinite(lower) and np.isfinite(upper)):
        sys.exit("Validation test failed at case %d: %s gives the bounds [%r, %r] on the gain." % (case, label, lower, upper))
    if not (lower - ROUNDING <= gain <= upper + ROUNDING):
        sys.exit("Validation test failed at case %d: %s gives the bounds [%.15g, %.15g], which do not contain %s, %.15g." % (case, label, lower, upper, what, gain))


def randommodel(seed):
    '''A random unichain and aperiodic model, as rewards[s][a] and transitions[s][a] = {next state: probability}.'''
    rng = np.random.default_rng(seed)
    rewards, transitions = [], []
    for s in range(NSTATES):
        rewards.append(list(rng.uniform(0.0, 10.0, NACTIONS)))
        state = []
        for a in range(NACTIONS):
            others = rng.choice(np.arange(1, NSTATES), size=NNEXT - 1, replace=False)
            probs = MINPROB + (1.0 - NNEXT * MINPROB) * rng.dirichlet(np.ones(NNEXT))
            state.append({0: float(probs[0]), **{int(j): float(p) for j, p in zip(others, probs[1:])}})
        transitions.append(state)
    return rewards, transitions


def policygain(rewards, transitions, policy):
    '''The gain of a unichain policy, from h + g = r + P h with h(0) = 0.'''
    R, P = matrices(rewards, transitions)
    n = len(R)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = np.eye(n) - np.array([P[s][policy[s]] for s in range(n)])
    A[:n, n] = 1.0
    A[n, 0] = 1.0
    return np.linalg.solve(A, np.r_[[R[s][policy[s]] for s in range(n)], 0.0])[n]


def raises(error, function):
    '''Whether the function raises the error; returns the message, or None.'''
    try:
        function()
    except error as raised:
        return str(raised)
    return None


# ------------------------------------------------------------------
# CASE 1: The hub-and-spoke problem, whose gain is known
# ------------------------------------------------------------------

for algorithmlabel, algorithm in ALGORITHMS:
    for parallel in (False, True):
        for tolerance in TOLERANCES:
            label = "%s (parallel=%s, tolerance %g) on the hub-and-spoke problem" % (algorithmlabel, parallel, tolerance)
            mdl = solve(HUBREWARDS, HUBTRANSITIONS, algorithm=algorithm, tolerance=tolerance, parallel=parallel)
            if not mdl.hasConverged():
                sys.exit("Validation test failed at case 1: %s did not converge." % label)
            lower, middle, upper = bounds(mdl)
            checkbounds(1, label, lower, upper, HUBGAIN)
            if not upper - lower < tolerance:
                sys.exit("Validation test failed at case 1: %s gives bounds %.3e apart, which is not less than the tolerance." % (label, upper - lower))
            if middle != (lower + upper) / 2.0:
                sys.exit("Validation test failed at case 1: %s gives getGain() = %.15g, where the midpoint of the bounds is %.15g." % (label, middle, (lower + upper) / 2.0))
            if abs(middle - HUBGAIN) > tolerance / 2.0:
                sys.exit("Validation test failed at case 1: %s gives getGain() = %.15g, which is %.3e from the gain %.15g, more than half the tolerance." % (label, middle, abs(middle - HUBGAIN), HUBGAIN))


# ------------------------------------------------------------------
# CASE 2: Random unichain models, against exact gains
# ------------------------------------------------------------------

for seed in SEEDS:
    rewards, transitions = randommodel(seed)
    optimalgain = exactaverage(rewards, transitions)[1]
    for algorithmlabel, algorithm in ALGORITHMS:
        label = "%s (tolerance %g) on random model %d" % (algorithmlabel, LOOSE, seed)
        mdl = solve(rewards, transitions, algorithm=algorithm, tolerance=LOOSE, parallel=False)
        if not mdl.hasConverged():
            sys.exit("Validation test failed at case 2: %s did not converge." % label)
        lower, middle, upper = bounds(mdl)
        checkbounds(2, label, lower, upper, policygain(rewards, transitions, [int(a) for a in mdl.getPolicy()]), "the gain of the policy found")
        checkbounds(2, label, lower, upper, optimalgain, "the optimal gain")
        if not upper - lower < LOOSE:
            sys.exit("Validation test failed at case 2: %s gives bounds %.3e apart, which is not less than the tolerance." % (label, upper - lower))


# ------------------------------------------------------------------
# CASE 3: A periodic model, where solve() stops at a limit
# ------------------------------------------------------------------

for algorithmlabel, algorithm in ALGORITHMS[:2]:   # (modified policy iteration would take 100 sweeps per iteration to its limit)
    label = "%s on the three-state cycle" % algorithmlabel
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mdl = solve(CYCLEREWARDS, CYCLETRANSITIONS, algorithm=algorithm, tolerance=LOOSE, parallel=False)
    if mdl.hasConverged() or not any(issubclass(w.category, mdpsolver.ConvergenceWarning) for w in caught):
        sys.exit("Validation test failed at case 3: %s converged or gave no ConvergenceWarning, where the cycle should stop it at its limit." % label)
    lower, middle, upper = bounds(mdl)
    checkbounds(3, label, lower, upper, CYCLEGAIN)
    if abs(middle - CYCLEGAIN) < LOOSE:
        sys.exit("Validation test failed at case 3: %s gives the midpoint %.15g, which equals the gain, where the bounds [%.15g, %.15g] should be too far apart for that." % (label, middle, lower, upper))


# ------------------------------------------------------------------
# CASE 4: The bias, against known biases
# ------------------------------------------------------------------

problems = [("the hub-and-spoke problem", HUBREWARDS, HUBTRANSITIONS, HUBBIAS)]
for seed in SEEDS:
    rewards, transitions = randommodel(seed)
    problems.append(("random model %d" % seed, rewards, transitions, exactaverage(rewards, transitions)[2]))

for problemlabel, rewards, transitions, knownbias in problems:
    for algorithmlabel, algorithm in ALGORITHMS:
        label = "%s (tolerance %g) on %s" % (algorithmlabel, TIGHT, problemlabel)
        mdl = solve(rewards, transitions, algorithm=algorithm, tolerance=TIGHT, parallel=False)
        if not mdl.hasConverged():
            sys.exit("Validation test failed at case 4: %s did not converge." % label)
        values = mdl.getValueVector()
        for reference in range(len(rewards)):
            for state in range(len(rewards)):
                bias = mdl.getBias(state, referenceIndex=reference)
                if not isinstance(bias, float) or bias != values[state] - values[reference]:
                    sys.exit("Validation test failed at case 4: getBias(%d, referenceIndex=%d) after %s gives %r, where it should be the float value of state %d minus that of state %d, %r." % (state, reference, label, bias, state, reference, values[state] - values[reference]))
                difference = abs(bias - (knownbias[state] - knownbias[reference]))
                if difference > BIASTOL:
                    sys.exit("Validation test failed at case 4: getBias(%d, referenceIndex=%d) after %s differs by %.3e from the known bias, which is more than %.1e." % (state, reference, label, difference, BIASTOL))
        for state in range(len(rewards)):
            if mdl.getBias(state) != mdl.getBias(state, referenceIndex=0):
                sys.exit("Validation test failed at case 4: getBias(%d) after %s differs from getBias(%d, referenceIndex=0)." % (state, label, state))
        if mdl.getBias(np.int64(2), referenceIndex=np.int64(1)) != mdl.getBias(2, referenceIndex=1):
            sys.exit("Validation test failed at case 4: getBias() after %s gives another result with the NumPy integers 2 and 1 than with the integers." % label)


# ------------------------------------------------------------------
# CASE 5: Errors, the bound and the reference state
# ------------------------------------------------------------------

mdl = mdpsolver.model()
mdl.mdp(discount=DISCOUNT, **arrayinput(HUBREWARDS, HUBTRANSITIONS, bycolumn=True))
CALLS = (("getGain()", lambda: mdl.getGain()), ("getBias(1)", lambda: mdl.getBias(1)))
for call, function in CALLS:
    message = raises(RuntimeError, function)
    if message is None or "solve()" not in message:
        sys.exit("Validation test failed at case 5: %s before solve() gives %r, where it should raise a RuntimeError naming solve()." % (call, message))

quietly(lambda: mdl.solve(tolerance=LOOSE))
for call, function in CALLS:
    message = raises(RuntimeError, function)
    if message is None or "discounted" not in message:
        sys.exit("Validation test failed at case 5: %s after a run under the discounted criterion gives %r, where it should raise a RuntimeError naming the criterion." % (call, message))

# a run under the average criterion after the discounted one (the note that
# the model has been solved before is printed)
quietly(lambda: mdl.solve(tolerance=TIGHT, criterion="average"))
lower, middle, upper = bounds(mdl)
checkbounds(5, "a run under the average criterion after a discounted one", lower, upper, HUBGAIN)
biases = np.array([mdl.getBias(s) for s in range(len(HUBREWARDS))])
if float(np.max(np.abs(biases - HUBBIAS))) > BIASTOL:
    sys.exit("Validation test failed at case 5: getBias() after a run under the average criterion that followed a discounted one gives %s, where the known bias is %s." % (biases, HUBBIAS))
for name, expected in (("Lower", lower), ("LOWER", lower), ("upper", upper), ("Upper", upper), ("UPPER", upper), (None, middle)):
    if mdl.getGain(bound=name) != expected:
        sys.exit("Validation test failed at case 5: getGain(bound=%r) gives %r, where it should give %r." % (name, mdl.getGain(bound=name), expected))
for name in ("middle", "", "low", 1, True):
    if raises(ValueError, lambda: mdl.getGain(bound=name)) is None:
        sys.exit("Validation test failed at case 5: getGain(bound=%r) does not raise a ValueError." % (name,))
for index, error in ((-1, IndexError), (len(HUBREWARDS), IndexError), (1.0, TypeError), ("0", TypeError), (True, TypeError), (None, TypeError)):
    for call, function in (("getBias(%r)" % (index,), lambda: mdl.getBias(index)),
                           ("getBias(0, referenceIndex=%r)" % (index,), lambda: mdl.getBias(0, referenceIndex=index))):
        if raises(error, function) is None:
            sys.exit("Validation test failed at case 5: %s does not raise %s." % (call, error.__name__))

# and a run under the discounted criterion again
quietly(lambda: mdl.solve(tolerance=LOOSE))
for call, function in CALLS:
    if raises(RuntimeError, function) is None:
        sys.exit("Validation test failed at case 5: %s after a discounted run that followed an average one does not raise a RuntimeError." % call)


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
