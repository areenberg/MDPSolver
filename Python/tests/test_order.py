'''
UNIT TEST FOR CALLING THE METHODS OF A MODEL OBJECT IN THE WRONG ORDER.

A model object is used in three steps: mdp() defines the model, solve() solves
it, and the output methods (getPolicy(), getValueVector(), getAction(),
getValue(), getRuntime(), getIterations(), hasConverged(), printPolicy(),
printValueVector() and saveToFile()) report the solution. checkModel() needs
only mdp(). Before a solution exists, the solver
holds a placeholder policy and value vector, a single -1 each, which the output
methods used to return as if it were a solution: getPolicy() gave [-1] and
getValue(0) gave -1.0, also for a model of several states, and saveToFile()
wrote the row 0,-1. solve() before mdp() did nothing, without a word, and left
getIterations() with an uninitialised number. An initial policy handed to
solve() before mdp() was kept unchecked, returned by getPolicy() as if it were
a solution, and used by the next run even when it was too short for the model
defined in between, so that the solver wrote past its end.

Each of these calls now raises a RuntimeError that says which method to call
first, and leaves the model object as it was.

The test uses the hub-and-spoke problem of knownsolutions.py, whose solution is
known in closed form.

Sub-tests:
    Case 1: Every output method raises a RuntimeError naming itself, mdp()
            and solve() on a new model object, and naming itself and solve()
            after mdp(), also after a run of solve() rejected by its own
            checks. It prints nothing besides, and saveToFile() writes no
            file. The list of output methods is checked against the methods of
            the class, so that an output method added later cannot go untested
            unnoticed. checkModel() before mdp() raises a RuntimeError naming
            mdp().
    Case 2: solve() before mdp() raises a RuntimeError naming mdp(), with or
            without initial vectors and whatever its other arguments, and
            prints nothing besides. It leaves the object as new: mdp() and
            solve() then give the known solution, without the note that the
            model has been solved before, and with one action and one value per
            state, so that an initial policy of the wrong length given before
            mdp() was not kept.
    Case 3: After solve(), every output method works and agrees with the
            others: getAction() and getValue() with getPolicy() and
            getValueVector(), which are the known solution; printPolicy(),
            printValueVector() and the file of saveToFile() with them; and
            getIterations() and getRuntime() are a positive count and a time,
            and hasConverged() is True.
            A later run of solve() rejected by its own checks leaves the
            solution in place.

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import inspect
import os
import sys
import tempfile

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import densetranmat, hubdiscounted, hubproblem


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

DISCOUNT = 0.9
TOLERANCE = 1e-9           # the epsilon handed to solve()

HUBREWARDS, HUBTRANSITIONS = hubproblem()
NSTATES = len(HUBREWARDS)
HUBPOLICY, HUBVALUES = hubdiscounted(DISCOUNT)
MODEL = dict(discount=DISCOUNT, rewards=HUBREWARDS, tranMatWithZeros=densetranmat(HUBTRANSITIONS))

# A placeholder policy of a single -1 is told apart from the solution only
# with more than one state, and a value of -1.0 only if no value is -1.0
if NSTATES < 2 or -1 in HUBPOLICY or np.any(HUBVALUES == -1.0):
    sys.exit("Validation test failed at initialization: the hub-and-spoke problem cannot be told apart from the placeholders of the solver.")

tmpdir = tempfile.TemporaryDirectory()

# The output methods, each with the arguments of a call
OUTPUTFILE = os.path.join(tmpdir.name, "policy.csv")
OUTPUTMETHODS = {
    "getRuntime": dict(),
    "getIterations": dict(),
    "hasConverged": dict(),
    "printPolicy": dict(),
    "printValueVector": dict(),
    "getAction": dict(stateIndex=1),
    "getValue": dict(stateIndex=1),
    "getPolicy": dict(),
    "getValueVector": dict(),
    "saveToFile": dict(fileName=OUTPUTFILE, type="policy"),
}

# Every public method of the class other than these is an output method
STEPMETHODS = {"initialize", "mdp", "solve", "checkModel"}
publicmethods = {name for name, _ in inspect.getmembers(mdpsolver.model, inspect.isfunction) if not name.startswith("_")}
if publicmethods - STEPMETHODS != set(OUTPUTMETHODS):
    sys.exit("Validation test failed at initialization: the output methods of the model class are %s, but the test covers %s." % (sorted(publicmethods - STEPMETHODS), sorted(OUTPUTMETHODS)))

# Case 2: the calls of solve() before mdp(), each with arguments it would
# otherwise accept or reject for other reasons
EARLYSOLVES = (
    ("solve()", dict()),
    ("solve() with value iteration", dict(algorithm="vi")),
    ("solve() with an initial policy too short for the model", dict(initPolicy=[1] * (NSTATES - 1))),
    ("solve() with an initial value vector too long for the model", dict(initValueVector=[5.0] * (NSTATES + 1))),
    ("solve() with a tolerance of 2", dict(tolerance=2.0)),
)


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def attempt(function):
    '''Runs function with everything printed captured; returns the exception it raised (None if it did not) and the output.'''
    def run():
        try:
            function()
        except Exception as error:
            return error
        return None
    return quietly(run)


def call(mdl, method):
    '''Calls an output method of mdl with its arguments; returns what it returns.'''
    return getattr(mdl, method)(**OUTPUTMETHODS[method])


def removeoutputfile():
    if os.path.exists(OUTPUTFILE):
        os.remove(OUTPUTFILE)


def checkrefused(case, label, mdl, method, named, unnamed):
    '''Exits unless the output method raises a RuntimeError that names the method and every call in named, and no call in unnamed.'''
    removeoutputfile()
    error, output = attempt(lambda: call(mdl, method))
    if error is None:
        sys.exit("Validation test failed at case %d: %s() %s was accepted." % (case, method, label))
    message = str(error)
    if type(error) is not RuntimeError or method + "()" not in message \
            or any(name not in message for name in named) or any(name in message for name in unnamed):
        sys.exit("Validation test failed at case %d: %s() %s raised %r, which is not a RuntimeError naming %s() and %s." % (case, method, label, error, method, " and ".join(named)))
    if output:
        sys.exit("Validation test failed at case %d: %s() %s printed %r besides raising an error." % (case, method, label, output.strip()))
    if os.path.exists(OUTPUTFILE):
        sys.exit("Validation test failed at case %d: %s() %s wrote a file." % (case, method, label))


def checksolution(case, label, policy, values):
    '''Exits unless the policy and the values are the known solution of the hub-and-spoke problem at DISCOUNT.'''
    if policy != HUBPOLICY:
        sys.exit("Validation test failed at case %d: %s gives the policy %s, where the known one is %s." % (case, label, policy, HUBPOLICY))
    values = np.array(values, dtype=float)
    if values.shape != (NSTATES,) or not np.all(np.isfinite(values)):
        sys.exit("Validation test failed at case %d: %s gives the values %s." % (case, label, values))
    difference = float(np.max(np.abs(values - HUBVALUES)))
    if difference > TOLERANCE:
        sys.exit("Validation test failed at case %d: %s gives values that differ by %.3e from the known values, which is more than %.1e." % (case, label, difference, TOLERANCE))


# ------------------------------------------------------------------
# CASE 1: Output methods before a solution
# ------------------------------------------------------------------

for method in OUTPUTMETHODS:
    mdl = mdpsolver.model()
    checkrefused(1, "on a new model object", mdl, method, ("mdp()", "solve()"), ())

    mdl.mdp(**MODEL)
    checkrefused(1, "after mdp()", mdl, method, ("solve()",), ("mdp()",))

    error, output = attempt(lambda: mdl.solve(tolerance=2.0))
    if type(error) is not ValueError or "tolerance" not in str(error):
        sys.exit("Validation test failed at case 1: solve() with the tolerance 2.0 gave %r, where it should raise a ValueError about the tolerance." % error)
    checkrefused(1, "after mdp() and a rejected run of solve()", mdl, method, ("solve()",), ("mdp()",))

# checkModel() needs a model, but no solution
error, output = attempt(lambda: mdpsolver.model().checkModel())
if type(error) is not RuntimeError or "mdp()" not in str(error) or output:
    sys.exit("Validation test failed at case 1: checkModel() on a new model object gave %r and printed %r, where it should raise a RuntimeError naming mdp() and print nothing else." % (error, output.strip()))


# ------------------------------------------------------------------
# CASE 2: solve() before mdp()
# ------------------------------------------------------------------

for label, given in EARLYSOLVES:
    mdl = mdpsolver.model()
    error, output = attempt(lambda: mdl.solve(**given))
    if error is None:
        sys.exit("Validation test failed at case 2: %s before mdp() was accepted." % label)
    if type(error) is not RuntimeError or "mdp()" not in str(error):
        sys.exit("Validation test failed at case 2: %s before mdp() raised %r, which is not a RuntimeError naming mdp()." % (label, error))
    if output:
        sys.exit("Validation test failed at case 2: %s before mdp() printed %r besides raising an error." % (label, output.strip()))

    # The object is as new: no initial vector was kept, and no run counted
    mdl.mdp(**MODEL)
    def run():
        mdl.solve(tolerance=TOLERANCE)
        return list(mdl.getPolicy()), list(mdl.getValueVector())
    (policy, values), output = quietly(run)
    if output:
        sys.exit("Validation test failed at case 2: solving the model defined after %s printed %r." % (label, output.strip()))
    checksolution(2, "solving the model defined after " + label, policy, values)


# ------------------------------------------------------------------
# CASE 3: Output methods after solve()
# ------------------------------------------------------------------

mdl = mdpsolver.model()
mdl.mdp(**MODEL)
quietly(lambda: mdl.solve(tolerance=TOLERANCE))

for rejected in (False, True):
    after = "after a rejected run of solve()" if rejected else "after solve()"
    if rejected:
        error, output = attempt(lambda: mdl.solve(tolerance=2.0))
        if type(error) is not ValueError or output:
            sys.exit("Validation test failed at case 3: solve() with the tolerance 2.0 after a run gave %r and printed %r, where it should raise a ValueError and print nothing else." % (error, output.strip()))

    # What each output method returns and prints; only the print methods print
    results, printed = {}, {}
    for method in OUTPUTMETHODS:
        removeoutputfile()
        try:
            results[method], printed[method] = quietly(lambda: call(mdl, method))
        except Exception as error:
            sys.exit("Validation test failed at case 3: %s() %s raised %r." % (method, after, error))
        if printed[method] and not method.startswith("print"):
            sys.exit("Validation test failed at case 3: %s() %s printed %r." % (method, after, printed[method].strip()))

    policy, values = list(results["getPolicy"]), np.array(results["getValueVector"], dtype=float)
    checksolution(3, "getPolicy() and getValueVector() " + after, policy, values)

    actions = [mdl.getAction(stateIndex=s) for s in range(NSTATES)]
    statevalues = [mdl.getValue(stateIndex=s) for s in range(NSTATES)]
    if actions != policy or not np.array_equal(statevalues, values):
        sys.exit("Validation test failed at case 3: getAction() and getValue() %s give %s and %s, where getPolicy() and getValueVector() give %s and %s." % (after, actions, statevalues, policy, list(values)))

    expected = ["%d: %d" % (s, a) for s, a in enumerate(policy)]
    if printed["printPolicy"].strip().splitlines() != expected:
        sys.exit("Validation test failed at case 3: printPolicy() %s printed %r, where it should print %r." % (after, printed["printPolicy"].strip(), "\n".join(expected)))
    lines = printed["printValueVector"].strip().splitlines()
    if len(lines) != NSTATES or any(not line.startswith("%d: " % s) for s, line in enumerate(lines)) \
            or not np.allclose([float(line.split(": ")[1]) for line in lines], values, rtol=1e-5):
        sys.exit("Validation test failed at case 3: printValueVector() %s printed %r, where it should print the %d values of getValueVector()." % (after, printed["printValueVector"].strip(), NSTATES))

    with open(OUTPUTFILE) as file:
        rows = file.read().strip().splitlines()
    expected = ["State_Index,Action_Index"] + ["%d,%d" % (s, a) for s, a in enumerate(policy)]
    if rows != expected:
        sys.exit("Validation test failed at case 3: saveToFile() %s wrote %r, where it should write %r." % (after, rows, expected))

    iterations, runtime = results["getIterations"], results["getRuntime"]
    if not isinstance(iterations, int) or iterations < 1:
        sys.exit("Validation test failed at case 3: getIterations() %s gives %r, where it should be a positive number of iterations." % (after, iterations))
    if not isinstance(runtime, float) or not runtime >= 0.0:
        sys.exit("Validation test failed at case 3: getRuntime() %s gives %r, where it should be a runtime in milliseconds." % (after, runtime))
    if results["hasConverged"] is not True:
        sys.exit("Validation test failed at case 3: hasConverged() %s gives %r, where the run converged." % (after, results["hasConverged"]))

tmpdir.cleanup()


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
