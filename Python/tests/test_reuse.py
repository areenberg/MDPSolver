'''
UNIT TEST FOR USING A MODEL OBJECT MORE THAN ONCE.

A model object holds one model, so mdp() can be called only once per model
object, and a second call raises a RuntimeError telling the user to create a
new model object. (A second model loaded into the same object made Python crash
when the transition probabilities were given elementwise or from a file, whose
loaders look for free places that the first model had filled, and left the
solver with parts of both models otherwise.)

solve() can be called any number of times. Every run after the first starts
from the policy and the value vector that the previous run left behind, unless
initPolicy or initValueVector replace them, and then prints a note that the
model has been solved before and that a new model object starts afresh. Value
iteration uses only the value vector.

The test uses the hub-and-spoke problem of knownsolutions.py, whose solution is
known in closed form.

Sub-tests:
    Case 1: A second call of mdp() raises a RuntimeError that tells the user
            to create a new model object, whichever format the first call gave the
            transition probabilities in, and whether or not the model was
            solved in between. The object keeps the first model: solving it
            gives the solution of that model at its discount factor. A call of
            mdp() rejected by its own checks (an invalid discount factor) does
            not count, so a valid call can follow it.
    Case 2: The first run of solve() prints nothing. Every later run prints the
            note when it takes anything from the previous run: the policy or
            the value vector under policy iteration and modified policy
            iteration, the value vector under value iteration, whichever
            initPolicy and initValueVector do not replace. With nothing taken
            from the previous run, it prints nothing. A run of solve()
            rejected by its own checks does not count and prints no note.
    Case 3: A second run with the same settings starts from the solution of the
            first, so it needs a single iteration (getIterations()) where the
            first needed more, and gives the same policy and values, under
            both criteria. A new model object starts afresh, as the note says.

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import os
import sys
import tempfile

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import densetranmat, elementwisetranmat, hubaverage, hubdiscounted, hubproblem


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

DISCOUNT = 0.9
OTHERDISCOUNT = 0.5        # the discount factor of the rejected second model in case 1
TOLERANCE = 1e-9           # the epsilon handed to solve()

# Under the average-reward criterion the differences v(s) - v(0) are compared
# with the bias; they lie within 1e-12 of it here
BIASTOL = 1e-8

ALGORITHMS = (("value iteration", "vi"), ("policy iteration", "pi"), ("modified policy iteration", "mpi"))
CRITERIA = ("discounted", "average")

HUBREWARDS, HUBTRANSITIONS = hubproblem()
NSTATES = len(HUBREWARDS)
HUBPOLICY, HUBVALUES = hubdiscounted(DISCOUNT)
AVERAGEPOLICY, AVERAGEGAIN, AVERAGEBIAS = hubaverage()

# A first and a second call of mdp() in case 1 differ in their discount
# factors, so that the values show which model the object holds
if np.allclose(hubdiscounted(OTHERDISCOUNT)[1], HUBVALUES, atol=1e-3):
    sys.exit("Validation test failed at initialization: the values of the hub-and-spoke problem at discount factors %g and %g are too close to tell the two apart." % (DISCOUNT, OTHERDISCOUNT))

# Case 2: whether a run of solve() after the first takes anything from the
# previous run, and so prints the note, by algorithm and by the initial vectors
# given (value iteration uses only the value vector)
GIVEN = {"no initial vectors": dict(),
         "initPolicy": dict(initPolicy=[0] * NSTATES),
         "initValueVector": dict(initValueVector=[0.0] * NSTATES),
         "initPolicy and initValueVector": dict(initPolicy=[0] * NSTATES, initValueVector=[0.0] * NSTATES)}
NOTED = {("vi", "no initial vectors"): True,
         ("vi", "initPolicy"): True,
         ("vi", "initValueVector"): False,
         ("vi", "initPolicy and initValueVector"): False}
for algorithm in ("pi", "mpi"):
    NOTED.update({(algorithm, "no initial vectors"): True,
                  (algorithm, "initPolicy"): True,
                  (algorithm, "initValueVector"): True,
                  (algorithm, "initPolicy and initValueVector"): False})


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def writefile(path, header, rows):
    '''Writes rows to a comma-separated file with a header line; repr() keeps every float exact.'''
    with open(path, "w", newline="") as file:
        file.write("\n".join([header] + [",".join(repr(x) for x in row) for row in rows]) + "\n")
    return path


def formats(directory):
    '''The hub-and-spoke problem in each format of the transition probabilities, with the reward formats taken in turn.'''
    elementwiserewards = [[s, a, reward] for s, state in enumerate(HUBREWARDS) for a, reward in enumerate(state)]
    return {
        "tranMatWithZeros": dict(rewards=HUBREWARDS, tranMatWithZeros=densetranmat(HUBTRANSITIONS)),
        "tranMatElementwise": dict(rewardsElementwise=elementwiserewards,
                                   tranMatElementwise=elementwisetranmat(HUBTRANSITIONS)),
        "tranMatProbs and tranMatColumns": dict(
            rewards=HUBREWARDS,
            tranMatProbs=[[list(probs.values()) for probs in state] for state in HUBTRANSITIONS],
            tranMatColumns=[[list(probs.keys()) for probs in state] for state in HUBTRANSITIONS]),
        "tranMatFromFile": dict(
            rewardsFromFile=writefile(os.path.join(directory, "rewards.csv"), "state,action,reward", elementwiserewards),
            tranMatFromFile=writefile(os.path.join(directory, "transitions.csv"), "stateFrom,action,stateTo,probability",
                                      elementwisetranmat(HUBTRANSITIONS))),
    }


def attempt(function):
    '''Runs function with everything printed captured; returns the exception it raised (None if it did not) and the output.'''
    def run():
        try:
            function()
        except Exception as error:
            return error
        return None
    return quietly(run)


def newmodel(arguments):
    '''A model object holding the hub-and-spoke problem at DISCOUNT.'''
    mdl = mdpsolver.model()
    mdl.mdp(discount=DISCOUNT, **arguments)
    return mdl


def solve(mdl, algorithm, criterion="discounted", **given):
    '''Solves the model held by mdl; returns the policy, the values, the number of iterations and anything printed.'''
    def run():
        mdl.solve(algorithm=algorithm, tolerance=TOLERANCE, criterion=criterion, **given)
        return list(mdl.getPolicy()), np.array(mdl.getValueVector(), dtype=float), mdl.getIterations()

    (policy, values, iterations), output = quietly(run)
    return policy, values, iterations, output


def isnote(output):
    '''Whether output is the note of one line that the model has been solved before and that a new model object starts afresh.'''
    return output.startswith("Note:") and "solved before" in output and "new model object" in output \
        and len(output.strip().splitlines()) == 1


def checksolution(case, label, policy, values, criterion="discounted"):
    '''Exits unless the policy and the values are the known solution of the hub-and-spoke problem at DISCOUNT.'''
    exactpolicy = HUBPOLICY if criterion == "discounted" else AVERAGEPOLICY
    if policy != exactpolicy:
        sys.exit("Validation test failed at case %d: %s gives the policy %s, where the known one is %s." % (case, label, policy, exactpolicy))
    if values.shape != (NSTATES,) or not np.all(np.isfinite(values)):
        sys.exit("Validation test failed at case %d: %s gives the values %s." % (case, label, values))
    if criterion == "discounted":
        difference, tolerance, what = float(np.max(np.abs(values - HUBVALUES))), TOLERANCE, "the known values"
    else:
        difference, tolerance, what = float(np.max(np.abs((values - values[0]) - AVERAGEBIAS))), BIASTOL, "the bias"
    if difference > tolerance:
        sys.exit("Validation test failed at case %d: %s gives values that differ by %.3e from %s, which is more than %.1e." % (case, label, difference, what, tolerance))


tmpdir = tempfile.TemporaryDirectory()
FORMATS = formats(tmpdir.name)


# ------------------------------------------------------------------
# CASE 1: mdp() once per model object
# ------------------------------------------------------------------

for firstlabel, first in FORMATS.items():
    for solvedfirst in (False, True):
        for secondlabel, second in FORMATS.items():
            label = "mdp() with %s after mdp() with %s%s" % (secondlabel, firstlabel, " and solve()" if solvedfirst else "")
            mdl = newmodel(first)
            if solvedfirst:
                solve(mdl, "vi")

            error, output = attempt(lambda: mdl.mdp(discount=OTHERDISCOUNT, **second))
            if error is None:
                sys.exit("Validation test failed at case 1: %s was accepted." % label)
            if type(error) is not RuntimeError or "new model object" not in str(error):
                sys.exit("Validation test failed at case 1: %s raised %r, which is not a RuntimeError telling to create a new model object." % (label, error))
            if output:
                sys.exit("Validation test failed at case 1: %s printed %r besides raising an error." % (label, output.strip()))

            # The object still holds the first model, at the first discount factor
            policy, values, iterations, output = solve(mdl, "vi", initValueVector=[0.0] * NSTATES)
            if output:
                sys.exit("Validation test failed at case 1: solving the model object after %s printed %r." % (label, output.strip()))
            checksolution(1, "solving the model object after " + label, policy, values)

# A call rejected by the checks of mdp() leaves the object free for a valid one
mdl = mdpsolver.model()
error, output = attempt(lambda: mdl.mdp(discount=1.5, **FORMATS["tranMatWithZeros"]))
if type(error) is not ValueError or "discount" not in str(error):
    sys.exit("Validation test failed at case 1: mdp() with the discount factor 1.5 gave %r, where it should raise a ValueError about the discount factor." % error)
error, output = attempt(lambda: mdl.mdp(discount=DISCOUNT, **FORMATS["tranMatElementwise"]))
if error is not None or output:
    sys.exit("Validation test failed at case 1: mdp() after a call rejected for its discount factor gave %r and printed %r, where it should define the model." % (error, output.strip()))
policy, values, iterations, output = solve(mdl, "vi")
if output:
    sys.exit("Validation test failed at case 1: solving the model defined after a rejected call of mdp() printed %r." % output.strip())
checksolution(1, "the model defined after a rejected call of mdp()", policy, values)


# ------------------------------------------------------------------
# CASE 2: The note on starting from the previous run
# ------------------------------------------------------------------

for algorithmlabel, algorithm in ALGORITHMS:
    for givenlabel, given in GIVEN.items():
        label = "%s with %s" % (algorithmlabel, givenlabel)
        mdl = newmodel(FORMATS["tranMatWithZeros"])

        output = solve(mdl, algorithm)[3]
        if output:
            sys.exit("Validation test failed at case 2: the first run of %s printed %r." % (algorithmlabel, output.strip()))

        # The second run, with the initial vectors given, and a third without
        # them, which takes the value vector from the previous run whatever the
        # algorithm
        for runlabel, rungiven, noted in (("a second run of " + label, given, NOTED[algorithm, givenlabel]),
                                          ("a third run of %s, after one with %s" % (algorithmlabel, givenlabel), {}, True)):
            output = solve(mdl, algorithm, **rungiven)[3]
            if not noted and output:
                sys.exit("Validation test failed at case 2: %s printed %r, where nothing is taken from the previous run." % (runlabel, output.strip()))
            if noted and not isnote(output):
                sys.exit("Validation test failed at case 2: %s printed %r, where it should print one note that the model has been solved before and that a new model object starts afresh." % (runlabel, output.strip()))

# A run rejected by the checks of solve() does not count as a run, before or after a real one
mdl = newmodel(FORMATS["tranMatWithZeros"])
for when in ("before", "after"):
    error, output = attempt(lambda: mdl.solve(algorithm="pi", tolerance=2.0))
    if type(error) is not ValueError or output:
        sys.exit("Validation test failed at case 2: solve() with the tolerance 2.0 %s the first run gave %r and printed %r, where it should raise a ValueError and print nothing else." % (when, error, output.strip()))
    if when == "before":
        output = solve(mdl, "pi")[3]
        if output:
            sys.exit("Validation test failed at case 2: the first run of solve() after a rejected one printed %r." % output.strip())


# ------------------------------------------------------------------
# CASE 3: The second run starts from the solution of the first
# ------------------------------------------------------------------

for criterion in CRITERIA:
    for algorithmlabel, algorithm in ALGORITHMS:
        label = "%s under the %s criterion" % (algorithmlabel, criterion)
        mdl = newmodel(FORMATS["tranMatWithZeros"])
        firstpolicy, firstvalues, firstiterations, output = solve(mdl, algorithm, criterion)
        checksolution(3, "the first run of " + label, firstpolicy, firstvalues, criterion)
        if firstiterations < 2:
            sys.exit("Validation test failed at case 3: the first run of %s needs %d iteration, so a second run starting from its solution could not need fewer." % (label, firstiterations))

        policy, values, iterations, output = solve(mdl, algorithm, criterion)
        if not isnote(output):
            sys.exit("Validation test failed at case 3: the second run of %s printed %r, where it should print the note that the model has been solved before." % (label, output.strip()))
        if iterations != 1:
            sys.exit("Validation test failed at case 3: the second run of %s needs %d iterations, where starting from the solution of the first (%d iterations) needs 1." % (label, iterations, firstiterations))
        checksolution(3, "the second run of " + label, policy, values, criterion)

        # A new model object starts from scratch, just as the first did
        policy, values, iterations, output = solve(newmodel(FORMATS["tranMatWithZeros"]), algorithm, criterion)
        if output or iterations != firstiterations or policy != firstpolicy or not np.array_equal(values, firstvalues):
            sys.exit("Validation test failed at case 3: %s on a new model object needs %d iterations and printed %r, where it should repeat the first run on the first object (%d iterations, nothing printed, the same results)." % (label, iterations, output.strip(), firstiterations))

tmpdir.cleanup()


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
