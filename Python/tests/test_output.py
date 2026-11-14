'''
UNIT TEST FOR THE OUTPUT OF A SOLVED MODEL: saveToFile(), getAction(),
getValue() AND THE REPORT ON CONVERGENCE.

saveToFile() writes the policy or the value vector to a NumPy .npy file
(filetype="npy", the default) or to a comma-separated file (filetype="csv");
until 0.12.0 it wrote only comma-separated files. type="value" used to write nothing, since the compiled module knew only
"values", and a file that could not be opened was reported on stderr only.
getAction() and getValue() used to return -1 and 0.0 for a state outside the
model, and 0.0 cannot be told from a real value; they now raise IndexError.

A run of solve() that does not converge used to be reported only with
verbose=True, or by a line printed by the compiled module that no program could
catch. It now issues a ConvergenceWarning, which says why, and hasConverged()
tells afterwards. A run has not converged when it stops at the limit of
1,000,000 iterations, when a policy evaluation of policy iteration stops at its
limit, when the values become infinite or NaN, or when the final check
(makeFinalCheck) finds a value outside the range that the rewards and the
discount allow.

Sub-tests:
    Case 1: saveToFile() with type="policy" and type="value" writes by
            default an .npy file that np.load reads as the array of
            getPolicy() (int64) or getValueVector() (float64), and with
            filetype="csv" a header and one row per state, equal to them (to
            the six significant digits written); both replace a file that is
            there. Without a file name, it writes result.npy, or result.csv
            with filetype="csv", in the working directory. An unknown type or
            filetype, and a name ending in .csv with filetype="npy" or in .npy
            with filetype="csv" (in any case), raise a ValueError, also in the
            compiled module, whose message for a name names the filetype to
            use; a file in a folder that does not exist raises a
            FileNotFoundError. None of them writes a file.
    Case 2: getAction() and getValue() agree with getPolicy() and
            getValueVector() in every state, and raise an IndexError naming the
            index for a state outside the model.
    Case 3: A run that converges issues no warning, and hasConverged() is
            True. Each way of not converging issues one ConvergenceWarning that
            says why, prints nothing besides, and leaves hasConverged() False
            and the results readable: the iteration limit (value iteration on a
            periodic chain under the average reward criterion, which stops
            after 1,000,000 iterations), the limit of a policy evaluation
            (policy iteration on that chain), values outside the possible range
            (probabilities that sum to 1.8), values that become infinite with
            makeFinalCheck=False, and uncorrected values with
            postProcessing=False, whose warning says so. The warning can be
            turned into an error, and is mdpsolver.ConvergenceWarning.
    Case 4: getPolicy() and getValueVector() return NumPy arrays, of int64
            and float64, with one entry per state: the solution. Each call
            returns a copy, so changing one changes neither the model nor what
            the methods return later, and a later solve() does not change an
            array returned before it. (They returned lists until 0.11.0.)

The models are given to mdp() as NumPy arrays. The hub-and-spoke problem,
whose states have different numbers of actions, is given in the elementwise
formats (rewardsElementwise and tranMatElementwise), with the rows of each
state-action pair in the order of tranMatWithZeros, so that the results are
those of the dense lists exactly; the small models of case 3 are plain arrays.
test_arrays.py checks that every format gives the same results as an array and
as a list.

mdpsolver is loaded by loadmdpsolver.py, which explains where the compiled
solvermodule is taken from.
'''

import os
import sys
import tempfile
import warnings

import numpy as np

from loadmdpsolver import mdpsolver, finalline, quietly
from knownsolutions import arrayinput, hubdiscounted, hubproblem


# ------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------

DISCOUNT = 0.9
TOLERANCE = 1e-9           # the epsilon handed to solve()
ITERATIONLIMIT = 1000000

HUBREWARDS, HUBTRANSITIONS = hubproblem()
NSTATES = len(HUBREWARDS)
HUBMODEL = arrayinput(HUBREWARDS, HUBTRANSITIONS, bycolumn=True)
HUBPOLICY, HUBVALUES = hubdiscounted(DISCOUNT)

# Case 3: the models and runs that do not converge, with the words their
# warning must contain. The chain alternates between its two states, so value
# iteration under the average reward criterion never settles. (parallel=False,
# since a parallel region per sweep would make 1,000,000 sweeps of two states
# slow.)
PERIODIC = dict(rewards=np.array([[1.0], [0.0]]), tranMatWithZeros=np.array([[[0.0, 1.0]], [[1.0, 0.0]]]))
SUMTO18 = dict(rewards=np.array([[5.0, 10.0], [-1.0, 1.0]]), tranMatWithZeros=np.array([[[0.9, 0.9], [0.1, 0.9]], [[0.4, 0.6], [0.8, 0.2]]]))
ONESTATE = dict(rewards=np.array([[1.0]]), tranMatWithZeros=np.array([[[1.0]]]))
NOTCONVERGED = (
    ("value iteration on a periodic chain under the average reward criterion", PERIODIC,
     dict(algorithm="vi", criterion="average", parallel=False), ("iteration limit of 1000000", "periodic")),
    ("policy iteration on a periodic chain under the average reward criterion", PERIODIC,
     dict(algorithm="pi", criterion="average", parallel=False), ("policy evaluation", "periodic")),
    ("probabilities that sum to 1.8", SUMTO18,
     dict(algorithm="vi"), ("outside the range", "checkModel()")),
    ("probabilities that sum to 1.8, with makeFinalCheck=False", SUMTO18,
     dict(algorithm="vi", makeFinalCheck=False), ("infinite or NaN", "checkModel()")),
    ("a model of one state, with postProcessing=False", ONESTATE,
     dict(algorithm="vi", tolerance=0.5, postProcessing=False, initValueVector=[0.0]), ("postProcessing=False", "makeFinalCheck=False")),
)

tmpdir = tempfile.TemporaryDirectory()


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def solved(arguments, **given):
    '''A model object with the model solved; returns it, the warnings issued, and anything printed.'''
    def run():
        mdl = mdpsolver.model()
        mdl.mdp(discount=DISCOUNT, **arguments)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mdl.solve(**{"tolerance": TOLERANCE, **given})
        return mdl, caught
    (mdl, caught), output = quietly(run)
    return mdl, [w for w in caught], output


def raises(function):
    '''The exception function raises (None if none), with everything printed captured, and the output.'''
    def run():
        try:
            function()
        except Exception as error:
            return error
        return None
    return quietly(run)


def readrows(path):
    with open(path) as file:
        return file.read().strip().splitlines()


def loadednpy(path):
    '''The array np.load reads from an .npy file, or the exception it raises.'''
    try:
        return np.load(path, allow_pickle=False)
    except Exception as error:
        return error


mdl, caught, output = solved(HUBMODEL)
if output or caught:
    sys.exit("Validation test failed at initialization: solving the hub-and-spoke problem printed %r and warned %r." % (output.strip(), [str(w.message) for w in caught]))
POLICY, VALUES = list(mdl.getPolicy()), list(mdl.getValueVector())
if POLICY != HUBPOLICY or float(np.max(np.abs(np.array(VALUES) - HUBVALUES))) > TOLERANCE:
    sys.exit("Validation test failed at initialization: the hub-and-spoke problem gives the policy %s, where the known one is %s." % (POLICY, HUBPOLICY))


# ------------------------------------------------------------------
# CASE 1: saveToFile()
# ------------------------------------------------------------------

for kind, header, column, dtype in (("policy", "State_Index,Action_Index", POLICY, np.int64), ("value", "State_Index,Value", VALUES, np.float64)):
    # The default filetype: the array of getPolicy() or getValueVector()
    path = os.path.join(tmpdir.name, "%s.npy" % kind)
    with open(path, "w") as file:
        file.write("this file is replaced\n")
    error, output = raises(lambda: mdl.saveToFile(fileName=path, type=kind))
    if error is not None or output:
        sys.exit("Validation test failed at case 1: saveToFile(type=%r) raised %r and printed %r." % (kind, error, output.strip()))
    array = loadednpy(path)
    if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != (NSTATES,) or not np.array_equal(array, column):
        sys.exit("Validation test failed at case 1: saveToFile(type=%r) wrote a file that np.load reads as %r, where it should be the %s array %r." % (kind, array, np.dtype(dtype).name, column))

    # filetype="csv": a header and a row per state
    path = os.path.join(tmpdir.name, "%s.csv" % kind)
    with open(path, "w") as file:
        file.write("this file is replaced\n")
    error, output = raises(lambda: mdl.saveToFile(fileName=path, type=kind, filetype="csv"))
    if error is not None or output:
        sys.exit("Validation test failed at case 1: saveToFile(type=%r, filetype='csv') raised %r and printed %r." % (kind, error, output.strip()))
    rows = readrows(path)
    if not rows or rows[0] != header or len(rows) != NSTATES + 1:
        sys.exit("Validation test failed at case 1: saveToFile(type=%r, filetype='csv') wrote %r, where it should write the header %r and %d rows." % (kind, rows, header, NSTATES))
    for s, row in enumerate(rows[1:]):
        index, entry = row.split(",")
        if int(index) != s or not np.isclose(float(entry), column[s], rtol=1e-5, atol=0.0):
            sys.exit("Validation test failed at case 1: saveToFile(type=%r, filetype='csv') wrote the row %r for state %d, where its %s is %r." % (kind, row, s, kind, column[s]))

# Without a file name: result.npy or result.csv in the working directory
folder = os.path.join(tmpdir.name, "defaults")
os.mkdir(folder)
here = os.getcwd()
os.chdir(folder)
try:
    results = [raises(lambda: mdl.saveToFile(type="value")), raises(lambda: mdl.saveToFile(type="value", filetype="csv"))]
finally:
    os.chdir(here)
for (error, output), given in zip(results, ("", ", filetype='csv'")):
    if error is not None or output:
        sys.exit("Validation test failed at case 1: saveToFile(type='value'%s) without a file name raised %r and printed %r." % (given, error, output.strip()))
if sorted(os.listdir(folder)) != ["result.csv", "result.npy"]:
    sys.exit("Validation test failed at case 1: saveToFile() without a file name wrote %s, where it should write result.npy, and result.csv with filetype='csv'." % sorted(os.listdir(folder)))
array, rows = loadednpy(os.path.join(folder, "result.npy")), readrows(os.path.join(folder, "result.csv"))
if not isinstance(array, np.ndarray) or not np.array_equal(array, VALUES) or rows[0] != "State_Index,Value" or len(rows) != NSTATES + 1:
    sys.exit("Validation test failed at case 1: saveToFile(type='value') without a file name wrote %r to result.npy and %r to result.csv, where they should hold the value vector." % (array, rows))

# Arguments that are not used, and a file that cannot be opened (the compiled
# module makes the same checks of the arguments)
for label, given, expected, words in (
        ("an unknown type", dict(fileName="result.npy", type="values2"), ValueError, "'policy' or 'value'"),
        ("an unknown filetype", dict(fileName="result.npy", filetype="npz"), ValueError, "'npy' or 'csv'"),
        ("a name ending in .csv and the default filetype", dict(fileName="result.csv"), ValueError, "filetype='csv'"),
        ("a name ending in .CSV and filetype='npy'", dict(fileName="result.CSV", filetype="npy"), ValueError, "filetype='csv'"),
        ("a name ending in .npy and filetype='csv'", dict(fileName="result.npy", filetype="csv"), ValueError, "filetype='npy'"),
        ("a file in a folder that does not exist", dict(fileName=os.path.join("missing", "result.npy")), FileNotFoundError, "missing")):
    path = os.path.join(tmpdir.name, given["fileName"])
    methods = [("saveToFile()", mdl.saveToFile)] + ([("The compiled module's saveToFile()", mdl.mdl.saveToFile)] if expected is ValueError else [])
    for name, method in methods:
        error, output = raises(lambda: method(**{**given, "fileName": path}))
        if not isinstance(error, expected) or words not in str(error):
            sys.exit("Validation test failed at case 1: %s with %s raised %r, where it should raise a %s that says %r." % (name, label, error, expected.__name__, words))
        if output or os.path.exists(path):
            sys.exit("Validation test failed at case 1: %s with %s printed %r or wrote a file." % (name, label, output.strip()))


# ------------------------------------------------------------------
# CASE 2: getAction() and getValue()
# ------------------------------------------------------------------

for s in range(NSTATES):
    if mdl.getAction(stateIndex=s) != POLICY[s] or mdl.getValue(stateIndex=s) != VALUES[s]:
        sys.exit("Validation test failed at case 2: getAction() and getValue() of state %d give %r and %r, where getPolicy() and getValueVector() give %r and %r." % (s, mdl.getAction(stateIndex=s), mdl.getValue(stateIndex=s), POLICY[s], VALUES[s]))

for index in (-1, NSTATES, NSTATES + 5):
    for method in ("getAction", "getValue"):
        error, output = raises(lambda: getattr(mdl, method)(stateIndex=index))
        if not isinstance(error, IndexError) or str(index) not in str(error) or output:
            sys.exit("Validation test failed at case 2: %s(stateIndex=%d) gave %r and printed %r, where it should raise an IndexError naming the index." % (method, index, error, output.strip()))


# ------------------------------------------------------------------
# CASE 3: The report on convergence
# ------------------------------------------------------------------

if not mdl.hasConverged():
    sys.exit("Validation test failed at case 3: hasConverged() is False after a run that converged.")

for label, arguments, given, words in NOTCONVERGED:
    run, caught, output = solved(arguments, **given)
    messages = [str(w.message) for w in caught]
    if output:
        sys.exit("Validation test failed at case 3: %s printed %r." % (label, output.strip()))
    if len(caught) != 1 or caught[0].category is not mdpsolver.ConvergenceWarning \
            or any(word not in messages[0] for word in words):
        sys.exit("Validation test failed at case 3: %s warned %r, where it should issue one ConvergenceWarning naming %s." % (label, messages, " and ".join(words)))
    if run.hasConverged():
        sys.exit("Validation test failed at case 3: hasConverged() is True after %s." % label)
    if len(run.getValueVector()) != len(arguments["rewards"]) or len(run.getPolicy()) != len(arguments["rewards"]):
        sys.exit("Validation test failed at case 3: the results of %s cannot be read." % label)
    if given.get("criterion") == "average" and given["algorithm"] == "vi" and run.getIterations() != ITERATIONLIMIT:
        sys.exit("Validation test failed at case 3: %s stopped after %d iterations, where it should stop at the limit of %d." % (label, run.getIterations(), ITERATIONLIMIT))

# The warning can be turned into an error
def strict():
    mdl = mdpsolver.model()
    mdl.mdp(discount=DISCOUNT, **SUMTO18)
    with warnings.catch_warnings():
        warnings.simplefilter("error", mdpsolver.ConvergenceWarning)
        mdl.solve(algorithm="vi")
error, output = raises(strict)
if not isinstance(error, mdpsolver.ConvergenceWarning):
    sys.exit("Validation test failed at case 3: with ConvergenceWarning turned into an error, solve() gave %r." % error)

tmpdir.cleanup()


# ------------------------------------------------------------------
# CASE 4: The policy and the value vector as NumPy arrays
# ------------------------------------------------------------------

mdl, caught, output = solved(HUBMODEL)
policy, values = mdl.getPolicy(), mdl.getValueVector()
for name, array, dtype in (("getPolicy()", policy, np.int64), ("getValueVector()", values, np.float64)):
    if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != (NSTATES,):
        sys.exit("Validation test failed at case 4: %s returned %r, where it should return a NumPy array of %s with %d entries." % (name, array, np.dtype(dtype).name, NSTATES))
if policy.tolist() != HUBPOLICY or float(np.max(np.abs(values - HUBVALUES))) > TOLERANCE:
    sys.exit("Validation test failed at case 4: getPolicy() and getValueVector() give %s and %s, which are not the known solution." % (policy, values))

# Copies: changing them changes neither the model nor what the methods return later
expectedpolicy, expectedvalues = policy.copy(), values.copy()
policy[:] = -7
values[:] = np.nan
if not np.array_equal(mdl.getPolicy(), expectedpolicy) or not np.array_equal(mdl.getValueVector(), expectedvalues) \
        or mdl.getAction(stateIndex=0) != expectedpolicy[0] or mdl.getValue(stateIndex=0) != expectedvalues[0]:
    sys.exit("Validation test failed at case 4: changing the arrays that getPolicy() and getValueVector() returned changed the model's results.")

# A later solve() leaves the arrays returned before it as they were
heldpolicy, heldvalues = mdl.getPolicy(), mdl.getValueVector()
def resolve():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl.solve(algorithm="vi", tolerance=0.5, initValueVector=np.zeros(NSTATES))
quietly(resolve)
if np.array_equal(mdl.getValueVector(), expectedvalues):
    sys.exit("Validation test failed at case 4: the second solve() gave the same values as the first, so it cannot show whether arrays returned before it change.")
if not np.array_equal(heldpolicy, expectedpolicy) or not np.array_equal(heldvalues, expectedvalues):
    sys.exit("Validation test failed at case 4: a later solve() changed the arrays that getPolicy() and getValueVector() returned before it.")


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

print(finalline(__file__))
