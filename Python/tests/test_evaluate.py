'''
UNIT TEST FOR EVALUATING A POLICY WITHOUT IMPROVING IT (model.evaluate()).

evaluate() runs the evaluation step of policy iteration, v <- r + discount P v
with the actions of a fixed policy, until an update changes the values by less
than the tolerance, and never changes the policy. The policy is the one given,
or else the one of the last run of solve() or evaluate(). The output methods
then report the evaluation: getPolicy() the policy, getValueVector() its
values, getGain() and getBias() its gain and bias under the average reward
criterion.

What the results must satisfy. Under discounting, with standard updates and
post-processing, the values of evaluate() are a lower bound on the values of
the policy, less than the tolerance below them: if v' is an update of v, the
values of the policy lie between v' + discount/(1 - discount) min(v' - v) and
v' + discount/(1 - discount) max(v' - v) (Puterman, 1994, Theorem 6.6.3, with
the policy as the only one). solve() returns such a lower bound for the policy
it finds, so when evaluate() is given that policy, both value vectors lie
within the tolerance below the values of the policy, and less than the
tolerance apart. Under the average reward criterion, the smallest and largest
entry of v' - v bound the gain of the policy, and the bias is the difference of
two values, as after solve().

The exact values, gains and biases of a policy are computed by solving its
linear equations, so they share no arithmetic with MDPSolver's iterative
evaluation.

Sub-tests:
    Case 1: evaluate() of the policy found by solve(), under the discounted
            criterion: on the hub-and-spoke problem at three discount factors,
            the secretary problem and random models, after value iteration,
            policy iteration and modified policy iteration, at a loose and a
            tight tolerance, on a new model object. The policy is returned
            unchanged, the values lie less than the tolerance below the exact
            values of the policy (and not above them), and less than the
            tolerance from the values of solve(); the run converged, and
            getIterations() is 0, as evaluate() makes no policy improvement
            step.
    Case 2: The same under the average reward criterion, on the hub-and-spoke
            problem and random unichain models: the bounds of getGain()
            contain the exact gain of the policy and are less than the
            tolerance apart, getGain() lies less than the tolerance from that
            of solve(), and at the tight tolerance getBias() lies within
            BIASTOL of the exact bias of the policy, for every state and
            several reference states.
    Case 3: Policies that are not optimal are evaluated, not improved: random
            policies on the random models, with every update ('standard', 'gs'
            and 'sor') and with serial and parallel computation, under both
            criteria. The policy is returned unchanged and its values (gain
            and bias) are those of the policy, below the optimal values by more
            than the tolerance somewhere. A run of solve() on the same object
            then prints the note that a policy has been evaluated before,
            finds the optimal values, and reports its improvement steps in
            getIterations().
    Case 4: The policy and the starting values: evaluate() without a policy
            evaluates the policy of the last run of solve() or evaluate(); the
            results do not depend on earlier runs on the object (they are
            those of a new object, to the last bit); the policy may be a list,
            a tuple or a NumPy array of any integer type, also a view; from
            initValueVector equal to the exact values of the policy, the
            values stay at them to within rounding, even at the loose
            tolerance. getIterations() is 0 after evaluate(), also after a
            run of solve() on the object. The other output methods agree
            with getPolicy() and getValueVector(); getGain() and getBias()
            after a discounted evaluation raise a RuntimeError naming the
            criterion. verbose prints the progress, and nothing is printed
            without it.
    Case 5: Policies whose evaluation never converges under the average reward
            criterion, a periodic one (a cycle of three states) and one whose
            gain differs between states (two absorbing states): evaluate()
            stops at its limit with a ConvergenceWarning that names evaluate()
            and points at the line that called it, hasConverged() is False,
            and the bounds still contain every gain of the policy. A warning of
            solve() points at the line that called it too.
    Case 6: Errors: evaluate() before mdp() raises a RuntimeError naming
            mdp(), and evaluate() without a policy before solve() one naming
            solve(). A policy or an initial value vector of the wrong length,
            an action outside its state, numbers that are not integers, the
            wrong number of dimensions, and invalid settings raise a ValueError
            or a TypeError. A rejected call prints nothing and leaves the object
            as it was: no results before a first run, the policy and the values
            of the last run after one.

References:
    Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic
    Dynamic Programming. Wiley. https://doi.org/10.1002/9780470316887

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import os
import sys
import warnings

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import (arrayinput, exactaverage, exactdiscounted, hubproblem,
                            matrices, secretaryproblem)


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOLERANCES = (1e-3, 1e-9)       # a loose and a tight epsilon
LOOSE = TOLERANCES[0]
TIGHT = TOLERANCES[1]
ROUNDING = 1e-12                # the margin for rounding, relative to the largest value (at least 1)
BIASTOL = 1e-8                  # how far a bias may lie from the exact one at the tight tolerance
DISCOUNT = 0.9                  # the discount factor of the random models and the secretary problem
HUBDISCOUNTS = (0.5, 0.9, 0.99)

ALGORITHMS = (("value iteration", "vi"), ("policy iteration", "pi"), ("modified policy iteration", "mpi"))
UPDATES = (("standard", dict(update="standard")), ("Gauss-Seidel", dict(update="gs")),
           ("SOR", dict(update="sor", SORrelaxation=1.2)))

HUBREWARDS, HUBTRANSITIONS = hubproblem()
SECRETARYREWARDS, SECRETARYTRANSITIONS = secretaryproblem(DISCOUNT)

# Random models in which every action leads to state 0 with a probability of at
# least MINPROB, so every policy is unichain (state 0 is in its only recurrent
# class) and aperiodic (state 0 can stay in state 0)
NSTATES, NACTIONS, NNEXT = 40, 3, 4
MINPROB = 0.05
SEEDS = range(5)
NPOLICIES = 3                   # random policies per random model in case 3

# Case 5: a cycle 0 -> 1 -> 2 -> 0 with a reward in state 2 only (gain 1 in
# every state), and two absorbing states with the rewards 1 and 2 (gains 1 and 2)
CYCLEREWARDS = [[0.0], [0.0], [3.0]]
CYCLETRANSITIONS = [[{1: 1.0}], [{2: 1.0}], [{0: 1.0}]]
CYCLEGAINS = (1.0,)
TWOCHAINREWARDS = [[1.0], [2.0]]
TWOCHAINTRANSITIONS = [[{0: 1.0}], [{1: 1.0}]]
TWOCHAINGAINS = (1.0, 2.0)


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


def randompolicy(seed, rewards):
    '''A random policy (a NumPy array of int64) for a model with the given rewards.'''
    rng = np.random.default_rng(1000 + seed)
    return np.array([rng.integers(0, len(state)) for state in rewards], dtype=np.int64)


def policyvalues(rewards, transitions, policy, discount):
    '''The exact discounted values of a policy, from (I - discount P) v = r.'''
    R, P = matrices(rewards, transitions)
    n = len(R)
    Ppolicy = np.array([P[s][policy[s]] for s in range(n)])
    rpolicy = np.array([R[s][policy[s]] for s in range(n)])
    return np.linalg.solve(np.eye(n) - discount * Ppolicy, rpolicy)


def policygainbias(rewards, transitions, policy):
    '''The exact gain and bias (0 in state 0) of a unichain policy, from h + g = r + P h, h(0) = 0.'''
    R, P = matrices(rewards, transitions)
    n = len(R)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = np.eye(n) - np.array([P[s][policy[s]] for s in range(n)])
    A[:n, n] = 1.0
    A[n, 0] = 1.0
    solution = np.linalg.solve(A, np.r_[[R[s][policy[s]] for s in range(n)], 0.0])
    return solution[n], solution[:n]


RANDOMMODELS = [randommodel(seed) for seed in SEEDS]

# The random policies of case 3 are worse than the optimal ones by far more
# than the loose tolerance in some state, under both criteria, so a policy
# that evaluate() improved could not pass for the policy given
for seed, (rewards, transitions) in zip(SEEDS, RANDOMMODELS):
    optimalvalues = exactdiscounted(rewards, transitions, DISCOUNT)[1]
    optimalgain = exactaverage(rewards, transitions)[1]
    for k in range(NPOLICIES):
        policy = randompolicy(NPOLICIES * seed + k, rewards)
        if np.max(optimalvalues - policyvalues(rewards, transitions, policy, DISCOUNT)) < 100 * LOOSE \
                or optimalgain - policygainbias(rewards, transitions, policy)[0] < 100 * LOOSE:
            sys.exit("Validation test failed at initialization: random policy %d of random model %d is nearly optimal, so case 3 could not tell it from an improved one." % (k, seed))


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def newmodel(rewards, transitions, discount=DISCOUNT):
    '''A new model object holding the model, given as NumPy arrays.'''
    mdl = mdpsolver.model()
    mdl.mdp(discount=discount, **arrayinput(rewards, transitions, bycolumn=True))
    return mdl


def run(case, label, function):
    '''Runs a call of solve() or evaluate(); anything it prints or warns ends the test.'''
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, output = quietly(function)
    if output:
        sys.exit("Validation test failed at case %d: %s printed %r." % (case, label, output.strip()))
    if caught:
        sys.exit("Validation test failed at case %d: %s warned %r." % (case, label, str(caught[0].message)))
    return result


def scale(values):
    '''The margin for rounding of a value vector.'''
    return ROUNDING * max(1.0, float(np.max(np.abs(values))))


def checkpolicy(case, label, mdl, policy):
    '''Ends the test unless getPolicy() returns the policy given, unchanged.'''
    returned = mdl.getPolicy()
    if not np.array_equal(returned, np.asarray(policy)):
        state = int(np.flatnonzero(returned != np.asarray(policy))[0])
        sys.exit("Validation test failed at case %d: %s returned a policy that differs from the one evaluated, in state %d (action %d instead of %d)." % (case, label, state, returned[state], policy[state]))


def checklowerbound(case, label, values, exact, tolerance):
    '''Ends the test unless the values lie less than the tolerance below the exact values, and not above them.'''
    margin = scale(exact)
    above = float(np.max(values - exact))
    below = float(np.max(exact - values))
    if above > margin or below >= tolerance + margin:
        sys.exit("Validation test failed at case %d: %s gives values from %.3e below to %.3e above the exact values of the policy, where they should lie less than the tolerance %g below them." % (case, label, below, above, tolerance))


def checkbounds(case, label, mdl, gains, tolerance):
    '''Ends the test unless the bounds of getGain() contain every gain given (and, with a tolerance, are less than it apart).'''
    lower, upper = mdl.getGain(bound="lower"), mdl.getGain(bound="upper")
    margin = ROUNDING * max(1.0, abs(upper))
    for gain in gains:
        if not lower - margin <= gain <= upper + margin:
            sys.exit("Validation test failed at case %d: %s gives the bounds [%.15g, %.15g], which do not contain the gain %.15g of the policy." % (case, label, lower, upper, gain))
    if tolerance is not None and not upper - lower < tolerance:
        sys.exit("Validation test failed at case %d: %s gives bounds %.3e apart, which is not less than the tolerance %g." % (case, label, upper - lower, tolerance))


def raises(error, function):
    '''Whether the function raises exactly the error, printing nothing; returns the message, or None.'''
    def attempt():
        try:
            function()
        except Exception as raised:
            return raised
        return None
    raised, output = quietly(attempt)
    if output:
        sys.exit("Validation test failed at case 6: a rejected call printed %r." % output.strip())
    return str(raised) if type(raised) is error else None


# ------------------------------------------------------------------
# CASE 1: The policy of solve(), under the discounted criterion
# ------------------------------------------------------------------

problems = [("the hub-and-spoke problem at discount %g" % d, HUBREWARDS, HUBTRANSITIONS, d) for d in HUBDISCOUNTS]
problems.append(("the secretary problem", SECRETARYREWARDS, SECRETARYTRANSITIONS, DISCOUNT))
problems += [("random model %d" % seed, rewards, transitions, DISCOUNT) for seed, (rewards, transitions) in zip(SEEDS, RANDOMMODELS)]

for problemlabel, rewards, transitions, discount in problems:
    for algorithmlabel, algorithm in ALGORITHMS:
        for tolerance in TOLERANCES:
            label = "evaluate() of the policy of %s (tolerance %g) on %s" % (algorithmlabel, tolerance, problemlabel)
            solved = newmodel(rewards, transitions, discount)
            run(1, label, lambda: solved.solve(algorithm=algorithm, tolerance=tolerance))
            policy = solved.getPolicy()
            mdl = newmodel(rewards, transitions, discount)
            run(1, label, lambda: mdl.evaluate(policy=policy, tolerance=tolerance))
            checkpolicy(1, label, mdl, policy)
            values = mdl.getValueVector()
            checklowerbound(1, label, values, policyvalues(rewards, transitions, policy, discount), tolerance)
            difference = float(np.max(np.abs(values - solved.getValueVector())))
            if difference >= tolerance + scale(values):
                sys.exit("Validation test failed at case 1: %s gives values %.3e from those of solve(), which is not less than the tolerance." % (label, difference))
            if not mdl.hasConverged() or mdl.getIterations() != 0:
                sys.exit("Validation test failed at case 1: %s did not converge, or reports %d iterations, where it makes no policy improvement step." % (label, mdl.getIterations()))


# ------------------------------------------------------------------
# CASE 2: The policy of solve(), under the average reward criterion
# ------------------------------------------------------------------

problems = [("the hub-and-spoke problem", HUBREWARDS, HUBTRANSITIONS)]
problems += [("random model %d" % seed, rewards, transitions) for seed, (rewards, transitions) in zip(SEEDS, RANDOMMODELS)]

for problemlabel, rewards, transitions in problems:
    for algorithmlabel, algorithm in ALGORITHMS:
        for tolerance in TOLERANCES:
            label = "evaluate() of the policy of %s (average reward, tolerance %g) on %s" % (algorithmlabel, tolerance, problemlabel)
            solved = newmodel(rewards, transitions)
            run(2, label, lambda: solved.solve(algorithm=algorithm, tolerance=tolerance, criterion="average"))
            policy = solved.getPolicy()
            mdl = newmodel(rewards, transitions)
            run(2, label, lambda: mdl.evaluate(policy=policy, tolerance=tolerance, criterion="average"))
            checkpolicy(2, label, mdl, policy)
            gain, bias = policygainbias(rewards, transitions, policy)
            checkbounds(2, label, mdl, [gain], tolerance)
            if not abs(mdl.getGain() - solved.getGain()) < tolerance:
                sys.exit("Validation test failed at case 2: %s gives the gain %.15g, %.3e from the gain %.15g of solve(), which is not less than the tolerance." % (label, mdl.getGain(), abs(mdl.getGain() - solved.getGain()), solved.getGain()))
            if tolerance == TIGHT:
                for reference in (0, 1, len(rewards) - 1):
                    for state in range(len(rewards)):
                        difference = abs(mdl.getBias(state, referenceIndex=reference) - (bias[state] - bias[reference]))
                        if difference > BIASTOL:
                            sys.exit("Validation test failed at case 2: getBias(%d, referenceIndex=%d) after %s differs by %.3e from the exact bias of the policy, more than %.1e." % (state, reference, label, difference, BIASTOL))
            if not mdl.hasConverged():
                sys.exit("Validation test failed at case 2: %s did not converge." % label)


# ------------------------------------------------------------------
# CASE 3: Policies that are not optimal are evaluated, not improved
# ------------------------------------------------------------------

for seed, (rewards, transitions) in zip(SEEDS, RANDOMMODELS):
    optimalvalues = exactdiscounted(rewards, transitions, DISCOUNT)[1]
    for k in range(NPOLICIES):
        policy = randompolicy(NPOLICIES * seed + k, rewards)
        exact = policyvalues(rewards, transitions, policy, DISCOUNT)
        gain, bias = policygainbias(rewards, transitions, policy)
        for parallel in (False, True):
            for updatelabel, update in UPDATES:
                label = "evaluate() of random policy %d (%s updates, parallel=%s) on random model %d" % (k, updatelabel, parallel, seed)
                mdl = newmodel(rewards, transitions)
                run(3, label, lambda: mdl.evaluate(policy=policy, tolerance=LOOSE, parallel=parallel, **update))
                checkpolicy(3, label, mdl, policy)
                values = mdl.getValueVector()
                if update["update"] == "standard":
                    checklowerbound(3, label, values, exact, LOOSE)
                elif float(np.max(np.abs(values - exact))) >= LOOSE:
                    # (the supremum norm stops the Gauss-Seidel and SOR updates)
                    sys.exit("Validation test failed at case 3: %s gives values %.3e from the exact values of the policy, which is not less than the tolerance." % (label, float(np.max(np.abs(values - exact)))))
                if not np.max(optimalvalues - values) > 10 * LOOSE:
                    sys.exit("Validation test failed at case 3: %s gives values within %.3e of the optimal values, as if the policy had been improved." % (label, float(np.max(optimalvalues - values))))

            label = "evaluate() of random policy %d (average reward, parallel=%s) on random model %d" % (k, parallel, seed)
            mdl = newmodel(rewards, transitions)
            run(3, label, lambda: mdl.evaluate(policy=policy, tolerance=TIGHT, criterion="average", parallel=parallel))
            checkpolicy(3, label, mdl, policy)
            checkbounds(3, label, mdl, [gain], TIGHT)
            difference = max(abs(mdl.getBias(s) - bias[s]) for s in range(NSTATES))
            if difference > BIASTOL:
                sys.exit("Validation test failed at case 3: %s gives a bias %.3e from the exact bias of the policy, more than %.1e." % (label, difference, BIASTOL))

        # solve() on the same object starts from the policy evaluated, says so,
        # and finds the optimal values
        label = "solve() after evaluate() of random policy %d on random model %d" % (k, seed)
        mdl = newmodel(rewards, transitions)
        run(3, label, lambda: mdl.evaluate(policy=policy, tolerance=LOOSE))
        _, output = quietly(lambda: mdl.solve(algorithm="pi", tolerance=LOOSE))
        if not (output.startswith("Note:") and "evaluated" in output and "new model object" in output and len(output.strip().splitlines()) == 1):
            sys.exit("Validation test failed at case 3: %s printed %r, where it should print one note that a policy has been evaluated before." % (label, output.strip()))
        checklowerbound(3, label, mdl.getValueVector(), optimalvalues, LOOSE)
        if mdl.getIterations() < 1:
            sys.exit("Validation test failed at case 3: %s reports %d iterations." % (label, mdl.getIterations()))


# ------------------------------------------------------------------
# CASE 4: The policy and the starting values
# ------------------------------------------------------------------

rewards, transitions = RANDOMMODELS[0]
policy = randompolicy(0, rewards)
exact = policyvalues(rewards, transitions, policy, DISCOUNT)


def results(mdl):
    '''The policy, the values and the number of iterations of the last run.'''
    return mdl.getPolicy(), mdl.getValueVector(), mdl.getIterations()


def same(first, second):
    '''Whether two results are identical, to the last bit.'''
    return all(np.array_equal(a, b) for a, b in zip(first[:2], second[:2])) and first[2] == second[2]


fresh = newmodel(rewards, transitions)
run(4, "evaluate() on a new model object", lambda: fresh.evaluate(policy=policy, tolerance=TIGHT))
reference = results(fresh)

# The policy in other forms
for formlabel, form in (("a list", [int(a) for a in policy]), ("a tuple", tuple(int(a) for a in policy)),
                        ("an int32 array", policy.astype(np.int32)), ("a uint8 array", policy.astype(np.uint8)),
                        ("a view of every other entry", np.repeat(policy, 2)[::2])):
    mdl = newmodel(rewards, transitions)
    run(4, "evaluate() of the policy as %s" % formlabel, lambda: mdl.evaluate(policy=form, tolerance=TIGHT))
    if not same(results(mdl), reference):
        sys.exit("Validation test failed at case 4: evaluate() of the policy given as %s gives other results than as an int64 array." % formlabel)

# After solve(), evaluate() without a policy evaluates the policy found, from
# the same starting values as a new object (not from the values of solve())
mdl = newmodel(rewards, transitions)
run(4, "solve()", lambda: mdl.solve(tolerance=LOOSE))
solvedpolicy = mdl.getPolicy()
if mdl.getIterations() < 1:
    sys.exit("Validation test failed at case 4: solve() reports %d iterations." % mdl.getIterations())
run(4, "evaluate() after solve()", lambda: mdl.evaluate(tolerance=TIGHT))
if mdl.getIterations() != 0:
    sys.exit("Validation test failed at case 4: getIterations() after evaluate() that followed solve() gives %d, where evaluate() makes no policy improvement step." % mdl.getIterations())
other = newmodel(rewards, transitions)
run(4, "evaluate() of the policy of solve() on a new model object", lambda: other.evaluate(policy=solvedpolicy, tolerance=TIGHT))
if not same(results(mdl), results(other)):
    sys.exit("Validation test failed at case 4: evaluate() without a policy after solve() gives other results than evaluate() of the policy of solve() on a new model object.")

# After evaluate(policy=...), the same: and again, without a policy
run(4, "evaluate() of a policy after another run", lambda: mdl.evaluate(policy=policy, tolerance=TIGHT))
if not same(results(mdl), reference):
    sys.exit("Validation test failed at case 4: evaluate() of a policy after solve() and evaluate() on the same object gives other results than on a new model object.")
run(4, "evaluate() without a policy after evaluate()", lambda: mdl.evaluate(tolerance=TIGHT))
if not same(results(mdl), reference):
    sys.exit("Validation test failed at case 4: evaluate() without a policy after evaluate(policy=...) does not evaluate that policy again, from the same starting values.")

# Starting from the exact values, the first update reaches the tolerance and
# changes them only by rounding, while from the default start the values end
# measurably below them at the loose tolerance
STARTTOL = 100 * scale(exact)
mdl = newmodel(rewards, transitions)
run(4, "evaluate() from the default start", lambda: mdl.evaluate(policy=policy, tolerance=LOOSE))
defaultgap = float(np.max(np.abs(mdl.getValueVector() - exact)))
if not defaultgap > 100 * STARTTOL:
    sys.exit("Validation test failed at case 4: from the default start, the values end only %.3e from the exact values at the loose tolerance, so the start could not be told apart." % defaultgap)
for formlabel, form in (("an array", exact), ("a list", list(exact))):
    mdl = newmodel(rewards, transitions)
    run(4, "evaluate() from the exact values", lambda: mdl.evaluate(policy=policy, tolerance=LOOSE, initValueVector=form))
    gap = float(np.max(np.abs(mdl.getValueVector() - exact)))
    if gap > STARTTOL:
        sys.exit("Validation test failed at case 4: evaluate() starting from the exact values of the policy, given as %s, ends %.3e from them, where it should stay at them to within %.1e." % (formlabel, gap, STARTTOL))

# The other output methods
mdl = fresh
for state in range(NSTATES):
    if mdl.getAction(state) != reference[0][state] or mdl.getValue(state) != reference[1][state]:
        sys.exit("Validation test failed at case 4: after evaluate(), getAction(%d) or getValue(%d) disagrees with getPolicy() or getValueVector()." % (state, state))
if not (mdl.hasConverged() and mdl.getIterations() == 0 and mdl.getRuntime() >= 0.0):
    sys.exit("Validation test failed at case 4: after evaluate(), hasConverged() is %r, getIterations() %r and getRuntime() %r." % (mdl.hasConverged(), mdl.getIterations(), mdl.getRuntime()))
for call, function in (("getGain()", lambda: mdl.getGain()), ("getBias(1)", lambda: mdl.getBias(1))):
    message = raises(RuntimeError, function)
    if message is None or "discounted" not in message or "evaluate()" not in message:
        sys.exit("Validation test failed at case 4: %s after a discounted evaluate() gives %r, where it should raise a RuntimeError naming the criterion and evaluate()." % (call, message))

# verbose
mdl = newmodel(rewards, transitions)
_, output = quietly(lambda: mdl.evaluate(policy=policy, tolerance=TIGHT, verbose=True))
if not ("Evaluating the policy" in output and "Policy evaluated in" in output and "sweeps" in output):
    sys.exit("Validation test failed at case 4: evaluate(verbose=True) printed %r, which does not report the evaluation and its sweeps." % output.strip())
if not same(results(mdl), reference):
    sys.exit("Validation test failed at case 4: evaluate(verbose=True) gives other results than without verbose.")


# ------------------------------------------------------------------
# CASE 5: Evaluations that never converge (average reward criterion)
# ------------------------------------------------------------------

# (serially: each of the million sweeps to the limit would open a parallel region)

def warned(function):
    '''Runs the function quietly; returns the ConvergenceWarnings it issued.'''
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quietly(function)
    return [w for w in caught if issubclass(w.category, mdpsolver.ConvergenceWarning)]


for modellabel, rewards, transitions, gains in (("the periodic policy of the three-state cycle", CYCLEREWARDS, CYCLETRANSITIONS, CYCLEGAINS),
                                                ("the policy of two absorbing states with different gains", TWOCHAINREWARDS, TWOCHAINTRANSITIONS, TWOCHAINGAINS)):
    label = "evaluate() of %s" % modellabel
    mdl = newmodel(rewards, transitions)
    caught = warned(lambda: mdl.evaluate(policy=[0] * len(rewards), tolerance=LOOSE, criterion="average", parallel=False))
    if mdl.hasConverged() or len(caught) != 1 or "evaluate()" not in str(caught[0].message):
        sys.exit("Validation test failed at case 5: %s converged or gave no ConvergenceWarning naming evaluate(), where it should stop at its limit." % label)
    if not os.path.samefile(caught[0].filename, __file__):
        sys.exit("Validation test failed at case 5: the ConvergenceWarning of %s points at %s, not at the line that called evaluate()." % (label, caught[0].filename))
    checkbounds(5, label, mdl, gains, None)

mdl = newmodel(CYCLEREWARDS, CYCLETRANSITIONS)
caught = warned(lambda: mdl.solve(algorithm="vi", tolerance=LOOSE, criterion="average", parallel=False))
if len(caught) != 1 or "solve()" not in str(caught[0].message) or not os.path.samefile(caught[0].filename, __file__):
    sys.exit("Validation test failed at case 5: value iteration on the three-state cycle gave %s, where it should give one ConvergenceWarning naming solve() and pointing at the line that called it." % [(str(w.message), w.filename) for w in caught])


# ------------------------------------------------------------------
# CASE 6: Errors and the order of calls
# ------------------------------------------------------------------

rewards, transitions = RANDOMMODELS[0]
policy = randompolicy(0, rewards)

message = raises(RuntimeError, lambda: mdpsolver.model().evaluate(policy=policy))
if message is None or "mdp()" not in message:
    sys.exit("Validation test failed at case 6: evaluate() before mdp() gives %r, where it should raise a RuntimeError naming mdp()." % message)

mdl = newmodel(rewards, transitions)
message = raises(RuntimeError, lambda: mdl.evaluate())
if message is None or "solve()" not in message or "policy" not in message:
    sys.exit("Validation test failed at case 6: evaluate() without a policy before solve() gives %r, where it should raise a RuntimeError naming the policy and solve()." % message)

tooshort = policy[:-1]
outside = policy.copy()
outside[7] = NACTIONS
negative = policy.copy()
negative[3] = -1
REJECTED = (
    ("a policy one state too short", ValueError, dict(policy=tooshort)),
    ("a policy one state too long", ValueError, dict(policy=np.r_[policy, 0])),
    ("a policy with an action outside its state", ValueError, dict(policy=outside)),
    ("a policy with a negative action", ValueError, dict(policy=negative)),
    ("a policy of floats", TypeError, dict(policy=policy.astype(float))),
    ("a policy with a float in a list", TypeError, dict(policy=[0.0] + [int(a) for a in policy[1:]])),
    ("a policy of two dimensions", TypeError, dict(policy=policy.reshape(-1, 1))),
    ("a policy given as a string", TypeError, dict(policy="0")),
    ("a policy given as None", TypeError, dict(policy=None)),
    ("an initial value vector one state too short", ValueError, dict(policy=policy, initValueVector=np.zeros(NSTATES - 1))),
    ("an initial value vector of strings", TypeError, dict(policy=policy, initValueVector=["0"] * NSTATES)),
    ("a tolerance of 0", ValueError, dict(policy=policy, tolerance=0.0)),
    ("a tolerance of 1", ValueError, dict(policy=policy, tolerance=1.0)),
    ("an unknown update", ValueError, dict(policy=policy, update="jacobi")),
    ("an unknown criterion", ValueError, dict(policy=policy, criterion="total")),
    ("Gauss-Seidel updates under the average reward criterion", ValueError, dict(policy=policy, update="gs", criterion="average")),
    ("a relaxation of 2 with SOR updates", ValueError, dict(policy=policy, update="sor", SORrelaxation=2.0)),
)

# On a model object without results, a rejected call leaves it without any
for rejectedlabel, error, arguments in REJECTED:
    mdl = newmodel(rewards, transitions)
    if raises(error, lambda: mdl.evaluate(**arguments)) is None:
        sys.exit("Validation test failed at case 6: evaluate() with %s does not raise a %s." % (rejectedlabel, error.__name__))
    if raises(RuntimeError, lambda: mdl.getPolicy()) is None:
        sys.exit("Validation test failed at case 6: evaluate() with %s, rejected, left results on a new model object." % rejectedlabel)

# After a run, a rejected call leaves its results in place, also when only the
# initial value vector is wrong (the policy is checked and kept back with it)
mdl = newmodel(rewards, transitions)
run(6, "evaluate()", lambda: mdl.evaluate(policy=policy, tolerance=TIGHT))
before = results(mdl)
otherpolicy = randompolicy(1, rewards)
for rejectedlabel, error, arguments in REJECTED + (("another policy with an initial value vector too long", ValueError, dict(policy=otherpolicy, initValueVector=np.zeros(NSTATES + 1))),):
    if raises(error, lambda: mdl.evaluate(**arguments)) is None:
        sys.exit("Validation test failed at case 6: evaluate() with %s after a run does not raise a %s." % (rejectedlabel, error.__name__))
    if not same(results(mdl), before):
        sys.exit("Validation test failed at case 6: evaluate() with %s, rejected, changed the results of the run before." % rejectedlabel)


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
