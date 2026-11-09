'''
UNIT TEST FOR VALUE ITERATION (algorithm="vi" in model.solve()).

Value iteration is run on two problems whose solutions are known in closed form,
under every setting of solve() that applies to it, and its policy and value
vector are compared with the known solutions.

The hub-and-spoke problem. In the hub (state 0) one either rests, for a reward
REST per period, or starts one of three ventures, which succeeds with
probability 1 - FAIL. Venture k (state k) pays R_k per period and ends with
probability q_k per period, back in the hub; two of the ventures can also be
quit, for nothing, back to the hub. All rewards are non-negative. With discount
factor lambda, the hub is worth m/(1 - lambda), where m is the best of REST and
lambda(1 - FAIL)R_k/(1 + lambda(q_k - FAIL)), the per-period worth of resting or
of venture k; a venture is continued if R_k >= lambda(1 - q_k)m, and is then
worth (R_k + lambda q_k v_hub)/(1 - lambda(1 - q_k)), and is quit otherwise,
worth lambda v_hub. The constants are chosen so that the best choice in the hub
moves from resting, to the short venture, to the long one as lambda grows, and
the poor venture goes from continued to quit. Under the average-reward criterion
the gain is the best of REST and (1 - FAIL)R_k/(1 + q_k - FAIL) (a renewal
argument), and with the bias of the hub set to 0, a continued venture has bias
(R_k - g)/q_k and a quit one -g.

The secretary problem (Ferguson, 1989). NCANDIDATES candidates are seen one at
a time in random order, and each can be selected, ending the search, or passed
over for good; the aim is to select the best of all. The optimal rule passes
over the first CUTOFF - 1 candidates and then selects the first one that is
better than all before it, which wins with probability about 1/e (0.398690 for
ten candidates). The states are a candidate's position i and whether it is the
best so far, plus an absorbing state once a candidate is selected or none is
left. The classical problem is undiscounted and MDPSolver discounts every step,
but every candidate takes exactly one step, so a reward of (i/n)/lambda^(i - 1)
for selecting candidate i undoes the discount exactly: the start state is worth
the classical probability of winning at every lambda, and candidate i's states
are worth their classical values divided by lambda^(i - 1).

Both closed forms are checked against the optimality equations before they are
used, and so are the margins by which the optimal actions win, so that a policy
comparison cannot hinge on rounding.

Sub-tests:
    Case 1: Infeasible arguments abort with an error message: an unknown
            algorithm, update method or optimality criterion, a tolerance
            outside (0, 1), an SOR relaxation outside (0, 2), an initial
            policy or value vector of the wrong length, and Gauss-Seidel or
            SOR updates under the average-reward criterion (see below).
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

The compiled solvermodule is found as in test_imports.py: from Python/src/
mdpsolver, CPP_Source_Code/build (unless older than the C++ sources), or the
installed package, in that order. It must have getNumberOfStates(), which the
checks of the initial policy and value vector use.

References:
    Ferguson, T. S. (1989). Who solved the secretary problem? Statistical
    Science 4(3), 282-289. https://doi.org/10.1214/ss/1177012493
    Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic
    Dynamic Programming. Wiley. https://doi.org/10.1002/9780470316887
'''

import ctypes
import glob
import importlib.machinery
import importlib.metadata
import importlib.util
import itertools
import os
import sys
import tempfile

import numpy as np

# Load mdpsolver from the src-folder so the test can be run without installing,
# with the compiled solvermodule from the first place that has one (see above)
TESTDIR = os.path.dirname(os.path.abspath(__file__))
SRCDIR = os.path.normpath(os.path.join(TESTDIR, "..", "src"))
CPPDIR = os.path.normpath(os.path.join(TESTDIR, "..", "..", "CPP_Source_Code"))
BUILDDIRS = (os.path.join(CPPDIR, "build"), os.path.join(CPPDIR, "build", "Release"))


def compiledmodules(directory):
    '''The files in a directory that this Python can load as solvermodule.'''
    paths = [os.path.join(directory, "solvermodule" + suffix)
             for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    return [path for path in paths if os.path.isfile(path)]


solvermodulenote = None   # said in the final line when the module is not in Python/src

if not os.path.isdir(os.path.join(SRCDIR, "mdpsolver")):
    solvermodulenote = "Python/src was not found next to the test, so the installed mdpsolver was tested as a whole."

elif not compiledmodules(os.path.join(SRCDIR, "mdpsolver")):
    sources = glob.glob(os.path.join(CPPDIR, "*.cpp")) + glob.glob(os.path.join(CPPDIR, "*.h"))
    newestsource = max(map(os.path.getmtime, sources), default=0.0)
    builds = [path for directory in BUILDDIRS for path in compiledmodules(directory)]
    currentbuilds = [path for path in builds if os.path.getmtime(path) >= newestsource]

    if currentbuilds:
        spec = importlib.util.spec_from_file_location("mdpsolver.solvermodule", currentbuilds[0])
        solvermodule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solvermodule)
        solvermodulenote = "The compiled solvermodule was loaded from %s." % currentbuilds[0]

    else:
        # The installed package is imported as a whole, before Python/src is on
        # the path, so that whatever it does to find the libraries its module
        # links to is done. Everything but its compiled module is then forgotten.
        try:
            import mdpsolver
        except ImportError:
            sys.exit("Validation test failed at initialization: there is no compiled solvermodule for this Python in Python/src/mdpsolver or in %s%s, and mdpsolver is not installed. Build it as described in the wiki's Contribution guidelines." % (BUILDDIRS[0], " (only one older than the C++ sources)" if builds else ""))
        solvermodule = mdpsolver.solvermodule
        for name in [name for name in sys.modules if name.split(".")[0] == "mdpsolver"]:
            del sys.modules[name]

        try:
            version = importlib.metadata.version("mdpsolver")
        except importlib.metadata.PackageNotFoundError:
            version = "(version unknown)"
        solvermodulenote = "The compiled solvermodule was that of the installed mdpsolver %s, %s, whose C++ may be older than that of the working tree." % (version, solvermodule.__file__)
        if builds:
            solvermodulenote += " The one in %s is older than the C++ sources and was passed over." % os.path.dirname(builds[0])

    # the Python code in Python/src imports the module as mdpsolver.solvermodule
    sys.modules["mdpsolver.solvermodule"] = solvermodule

sys.path.insert(0, SRCDIR)

import mdpsolver

if os.path.isdir(os.path.join(SRCDIR, "mdpsolver")) and not os.path.samefile(
        os.path.dirname(os.path.abspath(mdpsolver.__file__)), os.path.join(SRCDIR, "mdpsolver")):
    sys.exit("Validation test failed at initialization: mdpsolver was loaded from %s rather than from Python/src." % os.path.dirname(mdpsolver.__file__))

if not hasattr(mdpsolver.model().mdl, "getNumberOfStates"):
    sys.exit("Validation test failed at initialization: the compiled solvermodule has no getNumberOfStates(), so it is older than the Python code in Python/src. Rebuild it. (%s)" % (solvermodulenote or "It was loaded from Python/src/mdpsolver."))


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

TOLERANCE = 1e-9   # the epsilon handed to solve()

# The span stopping rule of the average-reward criterion bounds the error of
# the gain rather than of the bias, so the bias is compared at a tolerance set
# by measurement: the differences lie within 2e-10 of it here.
BIASTOL = 1e-8

# The optimal action of every state with more than one action must beat the
# runner-up by at least MINGAP, far more than the tolerance (the smallest gap
# is 1.7e-3 here).
MINGAP = 1e-4

# The hub-and-spoke problem: the reward for resting in the hub, the probability
# that starting a venture fails, and the ventures as (reward per period,
# probability of ending per period, whether it can be quit): a short one, a
# long one and a poor one.
REST = 2.0
FAIL = 0.1
VENTURES = ((6.0, 0.6, False),
            (4.2, 0.05, True),
            (1.5, 0.3, True))

# Discount factors for the hub-and-spoke problem, in increasing order, from
# where resting is best, past where the short venture is, to where the long one
# is and the poor one is quit
HUBDISCOUNTS = (0.3, 0.6, 0.8, 0.9, 0.95, 0.99)

# The discount factor in mdp() is not used under the average-reward criterion,
# but has to be valid
AVERAGEDISCOUNT = 0.5

# The secretary problem
NCANDIDATES = 10
SECRETARYDISCOUNTS = (0.5, 0.9, 0.999)
WINPROBABILITY = 0.398690   # for ten candidates (Ferguson, 1989), to six decimals

# Case 4: discount factors for the plain discounted secretary problem, in
# increasing order
PLAINDISCOUNTS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)

# An initial value vector far above every solution, for the warm starts
FARVALUE = 1e4

# The settings of solve() under which value iteration is run. Parallel
# computing is the default and is only available with standard updates. The
# initial value vectors are given by name and made to fit each problem.
DISCOUNTEDSETTINGS = (
    ("standard updates, in parallel", dict(update="standard")),
    ("standard updates, serially", dict(update="standard", parallel=False)),
    ("Gauss-Seidel updates", dict(update="gs")),
    ("SOR updates with relaxation 0.8", dict(update="sor", SORrelaxation=0.8)),
    ("SOR updates with relaxation 1.2", dict(update="sor", SORrelaxation=1.2)),
    ("SOR updates with relaxation 1.6", dict(update="sor", SORrelaxation=1.6)),
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
    ("an initial value vector one too short", dict(initValueVector=-1), "initial value vector"),
    ("an initial value vector one too long", dict(initValueVector=+1), "initial value vector"),
    ("an initial policy one too short", dict(initPolicy=-1), "initial policy"),
    ("an initial policy one too long", dict(initPolicy=+1), "initial policy"),
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
    ("an initial policy", dict(update="standard", initPolicy="zeros"), dict(update="standard")),
)
IGNOREDDISCOUNT = 0.9


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def hubproblem():
    '''The rewards[s][a] and transitions[s][a] = {next state: probability} of the hub-and-spoke problem.'''
    nventures = len(VENTURES)
    rewards = [[REST] + [0.0] * nventures]
    transitions = [[{0: 1.0}] + [{0: FAIL, k: 1.0 - FAIL} for k in range(1, nventures + 1)]]
    for k, (reward, end, quittable) in enumerate(VENTURES, start=1):
        rewards.append([reward] + ([0.0] if quittable else []))
        transitions.append([{0: end, k: 1.0 - end}] + ([{0: 1.0}] if quittable else []))
    return rewards, transitions


def hubdiscounted(discount):
    '''The optimal policy and values of the hub-and-spoke problem under discounting.'''
    worths = [REST] + [discount * (1.0 - FAIL) * reward / (1.0 + discount * (end - FAIL))
                       for reward, end, _ in VENTURES]
    best = max(worths)
    hub = best / (1.0 - discount)
    policy, values = [int(np.argmax(worths))], [hub]
    for reward, end, quittable in VENTURES:
        if quittable and reward < discount * (1.0 - end) * best:
            policy.append(1)
            values.append(discount * hub)
        else:
            policy.append(0)
            values.append((reward + discount * end * hub) / (1.0 - discount * (1.0 - end)))
    return policy, np.array(values)


def hubaverage():
    '''The gain-optimal policy, the gain and the bias (0 in the hub) of the hub-and-spoke problem.'''
    gains = [REST] + [(1.0 - FAIL) * reward / (1.0 + end - FAIL) for reward, end, _ in VENTURES]
    gain = max(gains)
    policy, bias = [int(np.argmax(gains))], [0.0]
    for reward, end, quittable in VENTURES:
        if quittable and reward < gain * (1.0 - end):
            policy.append(1)
            bias.append(-gain)
        else:
            policy.append(0)
            bias.append((reward - gain) / end)
    return policy, gain, np.array(bias)


def secretaryproblem(discount, compensated=True):
    '''
    The rewards and transitions of the secretary problem. State 2(i - 1) is
    candidate i being the best so far, state 2(i - 1) + 1 it not being so, and
    state 2n the end. A best-so-far candidate can be passed over (action 0) or
    selected (action 1); any other is passed over. Compensated, the reward for
    selecting candidate i undoes the discount of the i - 1 steps to it.
    '''
    n = NCANDIDATES
    end = 2 * n
    rewards, transitions = [], []
    for i in range(1, n + 1):
        onward = {end: 1.0} if i == n else {2 * i: 1.0 / (i + 1), 2 * i + 1: i / (i + 1)}
        rewards.append([0.0, (i / n) / (discount ** (i - 1) if compensated else 1.0)])
        transitions.append([onward, {end: 1.0}])
        rewards.append([0.0])
        transitions.append([dict(onward)])
    rewards.append([0.0])
    transitions.append([{end: 1.0}])
    return rewards, transitions


def secretarysolution(discount):
    '''
    The optimal policy and values of the secretary problem, from the classical
    solution: W(m) = (m/n) sum_{k=m}^{n-1} 1/k is the probability of winning by
    selecting the next best-so-far candidate after the m-th, and the cutoff is
    the first candidate from which selecting a best-so-far one is at least as
    good as going on.
    '''
    n = NCANDIDATES

    def W(m):
        return (m / n) * sum(1.0 / k for k in range(m, n))

    cutoff = min(r for r in range(1, n + 1) if sum(1.0 / k for k in range(r, n)) <= 1.0)
    policy, values = [], []
    for i in range(1, n + 1):
        onward = W(max(i, cutoff - 1))
        scale = discount ** (i - 1)
        policy += [1 if i >= cutoff else 0, 0]
        values += [max(i / n, onward) / scale, onward / scale]
    policy.append(0)
    values.append(0.0)
    return cutoff, policy, np.array(values)


def densetranmat(transitions):
    '''The transition probabilities as a 3D list with zeros.'''
    n = len(transitions)
    return [[[probs.get(j, 0.0) for j in range(n)] for probs in state] for state in transitions]


def qvalues(rewards, transitions, values, discount):
    '''r(s, a) + discount * sum_j p(j | s, a) v(j) for every state and action.'''
    return [[rewards[s][a] + discount * sum(p * values[j] for j, p in transitions[s][a].items())
             for a in range(len(rewards[s]))]
            for s in range(len(rewards))]


def checkoptimality(name, rewards, transitions, policy, values, discount, gain=0.0):
    '''
    Ends the test unless the values solve the optimality equation
    v + g = max_a (r + discount P v), with the policy as the maximizer and
    every maximum ahead of the runner-up by MINGAP.
    '''
    q = qvalues(rewards, transitions, values, discount)
    scale = max(1.0, float(np.max(np.abs(values))))
    for s in range(len(rewards)):
        if abs(values[s] + gain - max(q[s])) > 1e-12 * scale:
            sys.exit("Validation test failed at initialization: the known solution of the %s does not solve the optimality equation in state %d." % (name, s))
        if int(np.argmax(q[s])) != policy[s]:
            sys.exit("Validation test failed at initialization: the known policy of the %s does not maximize in state %d." % (name, s))
        if len(q[s]) > 1 and sorted(q[s])[-1] - sorted(q[s])[-2] < MINGAP:
            sys.exit("Validation test failed at initialization: in the %s the best action of state %d wins by only %.3e, so the comparison of policies would hinge on rounding." % (name, s, sorted(q[s])[-1] - sorted(q[s])[-2]))


def unichainaperiodic(rewards, transitions):
    '''The first policy that is not unichain and aperiodic, or None.'''
    n = len(rewards)
    for policy in itertools.product(*[range(len(r)) for r in rewards]):
        step = np.zeros((n, n), dtype=np.int64)
        for s in range(n):
            for j, p in transitions[s][policy[s]].items():
                step[s, j] = p > 0.0
        reach = np.eye(n, dtype=np.int64)
        for _ in range(n):
            reach = np.minimum(reach + reach @ step, 1)
        recurrent = [s for s in range(n) if all(reach[j, s] for j in range(n) if reach[s, j])]
        classes = {frozenset(j for j in recurrent if reach[s, j] and reach[j, s]) for s in recurrent}
        if len(classes) != 1:
            return policy
        members = sorted(next(iter(classes)))
        power = sub = step[np.ix_(members, members)]
        for _ in range(len(members) ** 2):   # primitive: some power is positive
            power = np.minimum(power @ sub, 1)
        if not np.all(power > 0):
            return policy
    return None


try:
    LIBC = ctypes.CDLL(None)   # the C library, to flush the output of the compiled module
except (OSError, TypeError):
    LIBC = None


def flushall():
    sys.stdout.flush()
    sys.stderr.flush()
    if LIBC is not None:
        LIBC.fflush(None)


def quietly(function):
    '''Runs a function with everything the process prints captured; returns its result and the output.'''
    flushall()
    with tempfile.TemporaryFile() as capture:
        saved = (os.dup(1), os.dup(2))
        os.dup2(capture.fileno(), 1)
        os.dup2(capture.fileno(), 2)
        try:
            result = function()
        finally:
            flushall()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        capture.seek(0)
        output = capture.read().decode(errors="replace")
    return result, output


def arguments(setting, n):
    '''The arguments of solve() for a setting, with the named initial vectors made to fit n states.'''
    result = dict(setting)
    named = {"zeros": [0.0] * n, "far": [FARVALUE] * n}
    if isinstance(result.get("initValueVector"), str):
        result["initValueVector"] = named[result["initValueVector"]]
    if isinstance(result.get("initPolicy"), str):
        result["initPolicy"] = [0] * n
    return result


def solve(rewards, transitions, discount, setting, criterion="discounted"):
    '''Value iteration under one setting; returns the policy, the value vector and anything printed.'''
    def run():
        mdl = mdpsolver.model()
        mdl.mdp(discount=discount, rewards=rewards, tranMatWithZeros=densetranmat(transitions))
        mdl.solve(algorithm="vi", tolerance=TOLERANCE, criterion=criterion, **arguments(setting, len(rewards)))
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
                    *hubsolutions[discount], discount)

averagepolicy, averagegain, averagebias = hubaverage()
checkoptimality("hub-and-spoke problem under the average-reward criterion", HUBREWARDS, HUBTRANSITIONS,
                averagepolicy, averagebias, 1.0, averagegain)

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
    checkoptimality("secretary problem at discount %g" % discount, rewards, transitions, policy, values, discount)
    secretarysolutions[discount] = (rewards, transitions, policy, values)

    if abs(values[0] - WINPROBABILITY) > 5e-7:
        sys.exit("Validation test failed at initialization: the start of the secretary problem is worth %.6f at discount %g, not the classical %.6f." % (values[0], discount, WINPROBABILITY))

if FARVALUE <= max(np.max(np.abs(solution[1])) for solution in hubsolutions.values()) or \
        FARVALUE <= max(np.max(np.abs(solution[3])) for solution in secretarysolutions.values()):
    sys.exit("Validation test failed at initialization: the far initial value vector is not above the solutions.")


# ------------------------------------------------------------------
# CASE 1: Infeasible arguments
# ------------------------------------------------------------------

nstates = len(HUBREWARDS)
for label, setting, word in INFEASIBLE:
    given = dict(algorithm="vi", tolerance=TOLERANCE)
    given.update(setting)
    if isinstance(given.get("initValueVector"), int):
        given["initValueVector"] = [0.0] * (nstates + given["initValueVector"])
    if isinstance(given.get("initPolicy"), int):
        given["initPolicy"] = [0] * (nstates + given["initPolicy"])

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

if solvermodulenote is None:
    print("vi: All tests passed.")
else:
    print("vi: All tests passed. (%s)" % solvermodulenote)
