'''
UNIT TEST FOR THE MODEL CHECKER, model.checkModel().

checkModel() looks for errors in the model that mdp() has defined, and returns
one message per kind of problem found, with how many there are and the first
few examples; an empty list when it finds none. It reads the whole model, so it
is never run by mdp() or solve(): solving a model costs no time on checks, and
the user calls checkModel() when the results look wrong or the model may be
malformed. A malformed model may make solve() crash or return garbage, but
checkModel() must report it without crashing itself.

The problems it finds: rewards and transition probabilities that cover
different numbers of states, or of actions in a state; states without actions;
state-action pairs without transition probabilities; tranMatProbs and
tranMatColumns of different shapes; next states outside the model;
probabilities that are not numbers between 0 and 1; probabilities of a
state-action pair that do not sum to 1 within a tolerance; and rewards that
are missing (left out of the elementwise formats or the files) or not finite.

Sub-tests:
    Case 1: A well-formed model, the hub-and-spoke problem of
            knownsolutions.py in every format of the transition probabilities,
            gives no problems, before and after solve(), and prints nothing.
            Checking changes nothing: the model then solves to its known
            solution.
    Case 2: Each kind of problem, in a small model and a format that can
            express it, gives exactly the expected messages, naming the first
            examples, and prints nothing besides.
    Case 3: The count and the examples: twenty state-action pairs whose
            probabilities do not sum to 1 give one message, which counts all
            twenty and shows the first three.
    Case 4: The tolerance: a sum that is off by 1e-7 is reported with the
            default tolerance (1e-9) and not with the tolerance 1e-6, and a sum
            off only by rounding (ten times 0.1) is not reported. A negative
            tolerance, or one that is not a number, raises a ValueError.
    Case 5: mdp() and solve() do not check the model: a model whose
            probabilities sum to 0.9 loads and solves without a message or a
            warning, and checkModel() reports it.

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import os
import sys
import tempfile
import warnings

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import densetranmat, elementwisetranmat, hubdiscounted, hubproblem


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

DISCOUNT = 0.9
TOLERANCE = 1e-9           # the epsilon handed to solve()

HUBREWARDS, HUBTRANSITIONS = hubproblem()
HUBPOLICY, HUBVALUES = hubdiscounted(DISCOUNT)

# The small model of case 2: two states with two actions each
R = [[5.0, 10.0], [-1.0, 1.0]]
P = [[[0.9, 0.1], [0.1, 0.9]], [[0.4, 0.6], [0.8, 0.2]]]
NAN, INF = float("nan"), float("inf")

# Case 2: each malformed model, as arguments of mdp(), with the messages that
# checkModel() must return
MALFORMED = (
    ("rewards for fewer states than the transition probabilities",
     dict(rewards=[[5.0, 10.0]], tranMatWithZeros=P),
     ["The rewards cover 1 state, but the transition probabilities cover 2."]),
    ("a state with fewer actions in the rewards",
     dict(rewards=[[5.0], [-1.0, 1.0]], tranMatWithZeros=P),
     ["1 state has a different number of actions in the rewards and in the transition probabilities: state 0 (1 in the rewards, 2 in the transition probabilities)."]),
    ("a state without actions",
     dict(rewards=[[1.0], []], tranMatWithZeros=[[[0.0, 1.0]], []]),
     ["1 state has no actions: state 1."]),
    ("tranMatProbs and tranMatColumns with different numbers of entries",
     dict(rewards=R, tranMatProbs=[[[0.9, 0.1], [1.0]], [[1.0], [1.0]]], tranMatColumns=[[[0, 1], [0, 1]], [[0], [1]]]),
     ["1 place where tranMatProbs and tranMatColumns differ in shape: state 0, action 1 (1 entry in tranMatProbs, 2 in tranMatColumns)."]),
    ("tranMatProbs and tranMatColumns with different numbers of states",
     dict(rewards=R, tranMatProbs=[[[1.0], [1.0]], [[1.0], [1.0]]], tranMatColumns=[[[0], [1]]]),
     ["tranMatProbs covers 2 states, but tranMatColumns covers 1."]),
    ("an action whose transition probabilities are all zero",
     dict(rewards=R, tranMatWithZeros=[[[0.9, 0.1], [0.0, 0.0]], [[0.4, 0.6], [0.8, 0.2]]]),
     ["1 state-action pair has no transition probabilities: state 0, action 1."]),
    ("a next state outside the model",
     dict(rewards=R, tranMatElementwise=[[0, 0, 0, 1.0], [0, 1, 7, 1.0], [1, 0, 1, 1.0], [1, 1, 0, 1.0]]),
     ["1 transition goes to a next state outside the model, whose states are 0 to 1: state 0, action 1, next state 7."]),
    ("probabilities below 0 and above 1",
     dict(rewards=R, tranMatElementwise=[[0, 0, 0, 1.5], [0, 0, 1, -0.5], [0, 1, 1, 1.0], [1, 0, 1, 1.0], [1, 1, 0, 1.0]]),
     ["2 transition probabilities are not numbers between 0 and 1: state 0, action 0, next state 0 (1.5); state 0, action 0, next state 1 (-0.5)."]),
    ("a probability that is NaN",
     dict(rewards=R, tranMatElementwise=[[0, 0, 0, NAN], [0, 1, 1, 1.0], [1, 0, 1, 1.0], [1, 1, 0, 1.0]]),
     ["1 transition probability is not a number between 0 and 1: state 0, action 0, next state 0 (nan)."]),
    ("probabilities that sum to 1.8",
     dict(rewards=R, tranMatWithZeros=[[[0.9, 0.9], [0.1, 0.9]], [[0.4, 0.6], [0.8, 0.2]]]),
     ["1 state-action pair has transition probabilities that do not sum to 1 (within 1e-09): state 0, action 0 (sum 1.8)."]),
    ("rewards that are NaN and infinite",
     dict(rewards=[[NAN, 10.0], [INF, 1.0]], tranMatWithZeros=P),
     ["2 state-action pairs have a reward that is missing or not a finite number: state 0, action 0 (missing or NaN); state 1, action 0 (inf)."]),
    ("a state left out of the elementwise formats",
     dict(rewardsElementwise=[[0, 0, 1.0], [2, 0, 1.0]], tranMatElementwise=[[0, 0, 2, 1.0], [2, 0, 0, 1.0]]),
     ["1 state-action pair has no transition probabilities: state 1, action 0.",
      "1 state-action pair has a reward that is missing or not a finite number: state 1, action 0 (missing or NaN)."]),
    # (the number of actions of a state is its highest action index plus one, so
    # leaving out its last action gives it one action fewer than the transitions)
    ("the first action of a state left out of rewardsElementwise",
     dict(rewardsElementwise=[[0, 1, 10.0], [1, 0, -1.0], [1, 1, 1.0]], tranMatWithZeros=P),
     ["1 state-action pair has a reward that is missing or not a finite number: state 0, action 0 (missing or NaN)."]),
    ("the last action of a state left out of rewardsElementwise",
     dict(rewardsElementwise=[[0, 0, 5.0], [1, 0, -1.0], [1, 1, 1.0]], tranMatWithZeros=P),
     ["1 state has a different number of actions in the rewards and in the transition probabilities: state 0 (1 in the rewards, 2 in the transition probabilities)."]),
)

# Case 3: twenty states with one action each, whose probabilities sum to 0.5
NBAD = 20
MANYBAD = dict(rewards=[[0.0]] * NBAD, tranMatWithZeros=[[[0.5] + [0.0] * (NBAD - 1)] for s in range(NBAD)])

if len(MALFORMED) < 10:
    sys.exit("Validation test failed at initialization: case 2 covers too few kinds of problems.")


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def newmodel(arguments, discount=DISCOUNT):
    '''A model object with the model given by the arguments of mdp().'''
    mdl = mdpsolver.model()
    mdl.mdp(discount=discount, **arguments)
    return mdl


def check(case, label, mdl, **given):
    '''checkModel() of mdl; exits if it raises an error or prints anything.'''
    try:
        problems, output = quietly(lambda: mdl.checkModel(**given))
    except Exception as error:
        sys.exit("Validation test failed at case %d: checkModel() on %s raised %r." % (case, label, error))
    if output:
        sys.exit("Validation test failed at case %d: checkModel() on %s printed %r." % (case, label, output.strip()))
    if not isinstance(problems, list) or not all(isinstance(problem, str) for problem in problems):
        sys.exit("Validation test failed at case %d: checkModel() on %s returned %r, where it should return a list of messages." % (case, label, problems))
    return problems


def writefile(path, header, rows):
    '''Writes rows to a comma-separated file with a header line; repr() keeps every float exact.'''
    with open(path, "w", newline="") as file:
        file.write("\n".join([header] + [",".join(repr(x) for x in row) for row in rows]) + "\n")
    return path


tmpdir = tempfile.TemporaryDirectory()


# ------------------------------------------------------------------
# CASE 1: A well-formed model
# ------------------------------------------------------------------

elementwiserewards = [[s, a, reward] for s, state in enumerate(HUBREWARDS) for a, reward in enumerate(state)]
WELLFORMED = {
    "rewards and tranMatWithZeros": dict(rewards=HUBREWARDS, tranMatWithZeros=densetranmat(HUBTRANSITIONS)),
    "rewardsElementwise and tranMatElementwise": dict(rewardsElementwise=elementwiserewards, tranMatElementwise=elementwisetranmat(HUBTRANSITIONS)),
    "tranMatProbs and tranMatColumns": dict(
        rewards=HUBREWARDS,
        tranMatProbs=[[list(probs.values()) for probs in state] for state in HUBTRANSITIONS],
        tranMatColumns=[[list(probs.keys()) for probs in state] for state in HUBTRANSITIONS]),
    "rewardsFromFile and tranMatFromFile": dict(
        rewardsFromFile=writefile(os.path.join(tmpdir.name, "rewards.csv"), "state,action,reward", elementwiserewards),
        tranMatFromFile=writefile(os.path.join(tmpdir.name, "transitions.csv"), "stateFrom,action,stateTo,probability",
                                  elementwisetranmat(HUBTRANSITIONS))),
}

for label, arguments in WELLFORMED.items():
    label = "the hub-and-spoke problem given as " + label
    mdl = newmodel(arguments)
    problems = check(1, label, mdl)
    if problems:
        sys.exit("Validation test failed at case 1: checkModel() on %s found %r." % (label, problems))

    (policy, values), output = quietly(lambda: (mdl.solve(tolerance=TOLERANCE), (list(mdl.getPolicy()), np.array(mdl.getValueVector())))[1])
    if output or policy != HUBPOLICY or values.shape != HUBVALUES.shape or float(np.max(np.abs(values - HUBVALUES))) > TOLERANCE:
        sys.exit("Validation test failed at case 1: solving %s after checkModel() gave the policy %s and printed %r, where the known solution is %s." % (label, policy, output.strip(), HUBPOLICY))

    problems = check(1, label + " after solve()", mdl)
    if problems:
        sys.exit("Validation test failed at case 1: checkModel() on %s after solve() found %r." % (label, problems))


# ------------------------------------------------------------------
# CASE 2: Each kind of problem
# ------------------------------------------------------------------

for label, arguments, expected in MALFORMED:
    problems = check(2, "a model with " + label, newmodel(arguments))
    if problems != expected:
        sys.exit("Validation test failed at case 2: checkModel() on a model with %s returned %r, where it should return %r." % (label, problems, expected))


# ------------------------------------------------------------------
# CASE 3: The count and the examples
# ------------------------------------------------------------------

problems = check(3, "twenty state-action pairs whose probabilities sum to 0.5", newmodel(MANYBAD))
expected = ["%d state-action pairs have transition probabilities that do not sum to 1 (within 1e-09), e.g. "
            "state 0, action 0 (sum 0.5); state 1, action 0 (sum 0.5); state 2, action 0 (sum 0.5)." % NBAD]
if problems != expected:
    sys.exit("Validation test failed at case 3: checkModel() returned %r, where it should return %r." % (problems, expected))


# ------------------------------------------------------------------
# CASE 4: The tolerance
# ------------------------------------------------------------------

offby = newmodel(dict(rewards=R, tranMatWithZeros=[[[0.9, 0.1 - 1e-7], [0.1, 0.9]], [[0.4, 0.6], [0.8, 0.2]]]))
problems = check(4, "a sum off by 1e-7", offby)
if len(problems) != 1 or "sum 0.9999999)" not in problems[0]:
    sys.exit("Validation test failed at case 4: checkModel() with the default tolerance returned %r for a sum off by 1e-7, where it should report the sum 0.9999999." % problems)
problems = check(4, "a sum off by 1e-7, with the tolerance 1e-6,", offby, tolerance=1e-6)
if problems:
    sys.exit("Validation test failed at case 4: checkModel(tolerance=1e-6) returned %r for a sum off by 1e-7." % problems)

# summed one by one, as checkModel() does (sum() compensates for rounding since Python 3.12)
naivesum = 0.0
for p in [0.1] * 10:
    naivesum += p
if naivesum == 1.0:
    sys.exit("Validation test failed at case 4: ten times 0.1 sums to exactly 1, so it cannot test a sum off by rounding.")
problems = check(4, "a sum off by rounding", newmodel(dict(rewards=[[1.0]], tranMatElementwise=[[0, 0, 0, 0.1]] * 10)))
if problems:
    sys.exit("Validation test failed at case 4: checkModel() returned %r for ten probabilities of 0.1, which sum to 1 up to rounding." % problems)

for tolerance in (-1e-9, "1e-9", NAN):
    try:
        quietly(lambda: offby.checkModel(tolerance=tolerance))
    except ValueError:
        pass
    except Exception as error:
        sys.exit("Validation test failed at case 4: checkModel(tolerance=%r) raised %r, where it should raise a ValueError." % (tolerance, error))
    else:
        sys.exit("Validation test failed at case 4: checkModel(tolerance=%r) was accepted." % (tolerance,))


# ------------------------------------------------------------------
# CASE 5: mdp() and solve() do not check the model
# ------------------------------------------------------------------

# The probabilities of every state-action pair sum to 0.9 (as if the process
# could end), which keeps the discounted values within the range of the rewards
SHORT = dict(rewards=R, tranMatWithZeros=[[[0.8, 0.1], [0.1, 0.8]], [[0.3, 0.6], [0.7, 0.2]]])

def loadandsolve():
    mdl = newmodel(SHORT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mdl.solve(tolerance=TOLERANCE)
    return mdl, caught

(mdl, caught), output = quietly(loadandsolve)
if output or caught:
    sys.exit("Validation test failed at case 5: loading and solving a model whose probabilities sum to 0.9 printed %r and warned %r, where the model is not checked unless checkModel() is called." % (output.strip(), [str(w.message) for w in caught]))
problems = check(5, "a model whose probabilities sum to 0.9", mdl)
if len(problems) != 1 or not problems[0].startswith("4 state-action pairs have transition probabilities that do not sum to 1"):
    sys.exit("Validation test failed at case 5: checkModel() on a model whose probabilities sum to 0.9 returned %r, where it should report the 4 state-action pairs." % problems)

tmpdir.cleanup()


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
