"""
MDPSolver

This file defines a `model` class that wraps around the `solvermodule` to create
and solve Markov Decision Processes (MDPs). It provides methods to configure the model,
solve it, and extract the resulting policy and value vectors.
"""

import functools
import numbers
import threading
import warnings

import numpy as np

from mdpsolver import solvermodule


class ConvergenceWarning(UserWarning):
    """
    Issued by solve() and evaluate() when a run has not converged.
    """

def _openError(fileName, mode="r"):
    """The OSError raised when the file is opened in the given mode, or None if it can be opened."""
    try:
        with open(fileName, mode):
            pass
    except OSError as error:
        return error
    return None


# The formats of mdp() as NumPy arrays: the number of dimensions and the shape (n = the
# number of rows)
_ARRAYSHAPES = {
    "rewards": (2, "states x actions"),
    "rewardsElementwise": (2, "n x 3: state, action, reward"),
    "tranMatWithZeros": (3, "states x actions x states"),
    "tranMatElementwise": (2, "n x 4: state, action, next state, probability"),
    "tranMatProbs": (3, "states x actions x next states per pair"),
    "tranMatColumns": (3, "states x actions x next states per pair"),
}


def _asInput(name, value, integer=False, toArray=False, keepFloat32=False):
    """
    An input format of mdp(): a list as it is (unless toArray), anything else as a
    C-contiguous NumPy array of float64 (int64 for integer) with the format's number of
    dimensions; with keepFloat32, an array of float32 stays float32 (the compiled module
    keeps such transition probabilities in single precision). An empty array is returned
    as an empty list, i.e. as not given.
    """
    if isinstance(value, list) and not toArray:
        return value
    if value is None or isinstance(value, (str, bytes)):
        raise TypeError("%s needs a list or a NumPy array, not %r." % (name, value))
    try:
        array = np.asarray(value)
    except ValueError as error:
        # e.g. a list whose rows differ in length
        raise ValueError("%s cannot be made a NumPy array (%s). Give it as an array of the shape %s, or give the format entirely as lists." % (name, error, _ARRAYSHAPES[name][1])) from None
    if array.dtype.kind not in "biuf":
        raise TypeError("%s needs numbers, but the array holds the type %s." % (name, array.dtype))
    ndim, shape = _ARRAYSHAPES[name]
    if array.ndim > 0 and len(array) == 0:
        return list()
    if array.ndim != ndim:
        raise ValueError("%s as a NumPy array needs %d dimensions (%s), but the array has %d." % (name, ndim, shape, array.ndim))
    if not integer:
        if keepFloat32 and array.dtype == np.float32:
            return np.ascontiguousarray(array)
        return np.ascontiguousarray(array, dtype=np.float64)
    if array.dtype.kind == "f":
        whole = np.isfinite(array) & (np.floor(array) == array)
        if not whole.all():
            raise ValueError("%s holds the column index %r, which is not an integer." % (name, float(array[~whole].flat[0])))
    return np.ascontiguousarray(array, dtype=np.int64)


def _asVector(label, value, integer):
    """
    A policy (integer) or a value vector given to solve() or evaluate(), as a 1D, C-contiguous
    NumPy array of int64 or float64, which the compiled module reads in place: an array that
    already is one is passed on as it is, without a copy, and a list is converted once. None
    when it is empty (not given). Raises TypeError for an entry that is not an integer (not a
    number); the compiled module checks the length and the action indices against the model,
    in one pass over the states (setInitialVectors()).
    """
    if value is None or isinstance(value, (str, bytes)):
        raise TypeError("The %s needs a list or a NumPy array, not %r." % (label, value))
    try:
        array = np.asarray(value)
    except (ValueError, TypeError):
        array = None    # e.g. entries of different lengths, found below
    if array is not None and array.ndim == 0:
        raise TypeError("The %s needs a list or a NumPy array, not %r." % (label, value))
    if array is not None and array.ndim == 1:
        if array.size == 0:
            return None
        # (booleans are integers in Python, as before)
        if array.dtype.kind in ("biu" if integer else "biuf"):
            return np.ascontiguousarray(array, dtype=np.int64 if integer else np.float64)
    # Otherwise, name the first entry that is not an integer (a number), as the checks
    # of every entry did before; only this path looks at the entries one by one
    entries = value.tolist() if hasattr(value, "tolist") else list(value)
    kind = numbers.Integral if integer else numbers.Real
    for state, entry in enumerate(entries):
        if not isinstance(entry, kind):
            raise TypeError("The %s needs %s for every state (state %d has %r)." % (label, "an integer action index" if integer else "a number", state, entry))
    # (numbers that NumPy keeps as Python objects)
    return np.array(entries, dtype=np.int64 if integer else np.float64)


def _checkSettings(tolerance, update, criterion):
    """Raise ValueError for a tolerance, update or criterion that solve() and evaluate() do not accept."""
    if tolerance <= 0.0 or tolerance >= 1.0:
        raise ValueError("The tolerance needs to be between 0.0 and 1.0 (i.e. 0 < tolerance < 1).")
    if not (update == "standard" or update == "gs" or update == "sor"):
        raise ValueError("Update method not recognized. Select either: 'standard', 'gs', or 'sor'.")
    if not (criterion == "discounted" or criterion == "average"):
        raise ValueError("Optimality criterion not recognized. Select either: 'discounted' or 'average'.")
    if criterion == "average" and (update == "gs" or update == "sor"):
        raise ValueError("Gauss-Seidel and SOR updates are not available with the average reward criterion. Select update='standard'.")


def _checkRelaxation(update, SORrelaxation):
    """Raise ValueError for a relaxation parameter outside (0, 2) with SOR updates."""
    if (SORrelaxation <= 0.0 or SORrelaxation >= 2.0) and update == "sor":
        raise ValueError("The relaxation parameter needs to be between 0.0 and 2.0 (i.e. 0 < SORrelaxation < 2).")


def _oneAtATime(method):
    """
    Runs a method of a model object with the object's lock held. solve() lets other
    Python threads run while it computes, so models in several threads are solved at
    the same time; the lock runs the calls on one model object one at a time, so that
    no call reads or changes the model while another thread solves it.
    """
    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


class model:
    """
    Creates an MDPSolver object.

    This class provides methods to initialize, solve, and return results
    from an MDP model.
    """

    def __init__(self):
        """Initialize the model by creating a solver module object."""
        self._lock = threading.RLock()
        self.initialize()

    @_oneAtATime
    def initialize(self):
        """Create and initialize the solver module object."""
        self.mdl = solvermodule.Model()
        # A model object contains only one model
        self._modelDefined = False
        # Whether solve() or evaluate() has run, so that the object holds results
        self._solved = False
        # The optimality criterion of the last run of solve() or evaluate(), and which of them ran
        self._criterion = None
        self._lastRun = None

    def _requireSolution(self, method):
        """
        Raise RuntimeError unless solve() or evaluate() has run on this model object.
        """
        if not self._modelDefined:
            raise RuntimeError("No model has been defined and solved for this model object, so %s() has no solution to use. Call mdp() and then solve() (or evaluate()) first." % method)
        if not self._solved:
            raise RuntimeError("The model has not been solved yet, so %s() has no solution to use. Call solve() (or evaluate()) first." % method)

    @_oneAtATime
    def solve(
        self,
        algorithm: str = "mpi",
        tolerance: float = 1e-3,
        update: str = "standard",
        criterion: str = "discounted",
        parIterLim: int = 100,
        SORrelaxation: float = 1.0,
        initPolicy: "list | numpy.ndarray" = list(),
        initValueVector: "list | numpy.ndarray" = list(),
        verbose: bool = False,
        postProcessing: bool = True,
        makeFinalCheck: bool = True,
        parallel: bool = True,
    ) -> None:
        """
        Derive an epsilon-optimal policy for the selected MDP model.

        Args:
            algorithm (str): Algorithm to use.
            tolerance (float): Convergence threshold for the algorithm (0 < tolerance < 1).
            update (str): The value-update method ('standard', 'gs' or 'sor'; only 'standard' with the average reward criterion).
            criterion (str): The optimality criterion.
            parIterLim (int): The partial evaluation limit employed in the modified policy iteration algorithm.
            SORrelaxation (float): Relaxation parameter for the Successive Over-Relaxation method.
            initPolicy (list or NumPy array): The initial policy, one valid action index per state. This option can be used for "warm starting" the optimization procedure.
            initValueVector (list or NumPy array): The initial value vector, one value per state. This option can be used for "warm starting" the optimization procedure.
            verbose (bool): If True, prints solver progress to console.
            postProcessing (bool): If True, performs post-processing after solving.
            makeFinalCheck (bool): If True, makes a final check of the value vector. This process checks if the resulting values are reasonable.
            parallel (bool): If True, enables parallel computation for faster solving. This option is only available for the custom MDP model with standard updates.

        Call mdp() to define the model first. Solving the same model object
        again starts from the policy and the value vector of the previous run,
        unless initPolicy or initValueVector replace them. Create a new model
        object to start over.

        Returns:
            None
        """
        
        if not self._modelDefined:
            raise RuntimeError("No model has been defined for this model object. Call mdp() before solve().")
        if not (algorithm == "mpi" or algorithm == "pi" or algorithm == "vi"):
            raise ValueError("Algorithm type not recognized. Select either: 'mpi', 'pi', or 'vi'.")
        _checkSettings(tolerance, update, criterion)
        if parIterLim <= 0 and algorithm == "mpi":
            raise ValueError("The partial evaluation limit needs to be a positive number.")
        _checkRelaxation(update, SORrelaxation)
        # NumPy arrays, read in place by the compiled module (None if not given)
        initValueVector = _asVector("initial value vector", initValueVector, integer=False)
        initPolicy = _asVector("initial policy", initPolicy, integer=True)

        # The compiled module checks the lengths and the actions against the model
        # (ValueError) before it changes anything, and sets the vectors
        if initPolicy is not None or initValueVector is not None:
            self.mdl.setInitialVectors(policy=initPolicy, initValueVector=initValueVector, label="initial policy", name="initPolicy")

        # The solver keeps the policy and the value vector of the previous run,
        # unless initPolicy or initValueVector replace them.

        if self._solved:
            # Value iteration never uses the policy it starts from
            reusesPolicy = initPolicy is None and algorithm != "vi"
            reusesValueVector = initValueVector is None
            if reusesPolicy or reusesValueVector:
                before = "This model has been solved before." if self._lastRun == "solve" else "A policy has been evaluated on this model before."
                print("Note: %s Create a new model object to run solver with initial values." % before, flush=True)

        self.mdl.solve(
            algorithm=algorithm,
            tolerance=tolerance,
            update=update,
            criterion=criterion,
            parIterLim=parIterLim,
            SORrelaxation=SORrelaxation,
            verbose=verbose,
            postProcessing=postProcessing,
            makeFinalCheck=makeFinalCheck,
            parallel=parallel,
        )
        # Set after the run, so that a run that did not finish leaves the
        # object without a solution
        self._solved = True
        self._criterion = criterion
        self._lastRun = "solve"
        self._warnConvergence("solve()", criterion, update, postProcessing)

    def _warnConvergence(self, method, criterion, update, postProcessing):
        """
        Issue a ConvergenceWarning if the run of solve() or evaluate() (the
        method) that just finished has not converged.
        """
        # (stacklevel 4 names the line that called solve() or evaluate(): this
        # method, the method, and the lock of _oneAtATime come first)
        convergence = self.mdl.getConvergence()
        hint = ""
        if criterion == "average" and method == "solve()":
            hint = " Under the average reward criterion, this happens with periodic models."
        elif criterion == "average":
            hint = " Under the average reward criterion, this happens with periodic policies, and with policies whose gain differs between states."
        if convergence["iterationLimitReached"] and method == "evaluate()":
            warnings.warn("evaluate() stopped at its limit of %d value updates before reaching the tolerance, so the results have not converged.%s" % (convergence["iterationLimit"], hint), ConvergenceWarning, stacklevel=4)
        elif convergence["iterationLimitReached"]:
            warnings.warn("%s stopped at the iteration limit of %d iterations before reaching the tolerance, so the results have not converged.%s" % (method, convergence["iterationLimit"], hint), ConvergenceWarning, stacklevel=4)
        elif convergence["evaluationLimitReached"]:
            warnings.warn("A policy evaluation of policy iteration stopped at its limit of %d iterations before reaching the tolerance, so the results have not converged.%s" % (convergence["iterationLimit"], hint), ConvergenceWarning, stacklevel=4)
        if convergence["valuesOutOfRange"] and not postProcessing and update == "standard":
            # The values are left uncorrected, and may lie outside the range
            # even when the policy is right
            warnings.warn("The final check found the value %r in state %d, outside the range that the rewards and the discount allow. With postProcessing=False, the values are not corrected and may lie outside it; use postProcessing=True for correct values, or makeFinalCheck=False to skip the check." % (convergence["outOfRangeValue"], convergence["outOfRangeState"]), ConvergenceWarning, stacklevel=4)
        elif convergence["valuesOutOfRange"]:
            warnings.warn("The final check found the value %r in state %d, outside the range that the rewards and the discount allow, so the results are wrong. checkModel() can look for errors in the model." % (convergence["outOfRangeValue"], convergence["outOfRangeState"]), ConvergenceWarning, stacklevel=4)
        elif convergence["diverged"]:
            warnings.warn("%s stopped because the values became infinite or NaN, so the results are wrong. checkModel() can look for errors in the model." % method, ConvergenceWarning, stacklevel=4)

    @_oneAtATime
    def evaluate(
        self,
        policy: "list | numpy.ndarray" = list(),
        tolerance: float = 1e-3,
        update: str = "standard",
        criterion: str = "discounted",
        SORrelaxation: float = 1.0,
        initValueVector: "list | numpy.ndarray" = list(),
        verbose: bool = False,
        postProcessing: bool = True,
        makeFinalCheck: bool = True,
        parallel: bool = True,
    ) -> None:
        """
        Evaluate a policy without improving it: compute its value vector,
        and under the average reward criterion its gain and bias.

        Args:
            policy (list or NumPy array): The policy to evaluate, one valid action index per state. By default the policy of the last run of solve() or evaluate().
            tolerance (float): Convergence threshold for the evaluation (0 < tolerance < 1).
            update (str): The value-update method ('standard', 'gs' or 'sor'; only 'standard' with the average reward criterion).
            criterion (str): The optimality criterion the policy is evaluated under: 'discounted' (with the discount given to mdp()) or 'average'.
            SORrelaxation (float): Relaxation parameter for the Successive Over-Relaxation method.
            initValueVector (list or NumPy array): The initial value vector, one value per state. By default the evaluation starts from the values a first run of solve() starts from.
            verbose (bool): If True, prints the progress of the evaluation to console.
            postProcessing (bool): If True, performs post-processing after evaluating.
            makeFinalCheck (bool): If True, makes a final check of the value vector. This process checks if the resulting values are reasonable.
            parallel (bool): If True, enables parallel computation for faster evaluation. This option is only available for the custom MDP model with standard updates.

        The values are updated with the actions of the policy, as in the
        evaluation step of policy iteration, until an update changes them by
        less than the tolerance. The output methods then report the
        evaluation: getPolicy() returns the policy evaluated,
        getValueVector() its values, and getGain() and getBias() its gain and
        bias. No policy improvement step is made, so getIterations() returns
        0. A later run of solve() starts from this policy and value vector.

        Returns:
            None
        """

        if not self._modelDefined:
            raise RuntimeError("No model has been defined for this model object. Call mdp() before evaluate().")
        # NumPy arrays, read in place by the compiled module (None if not given)
        _checkSettings(tolerance, update, criterion)
        _checkRelaxation(update, SORrelaxation)
        initValueVector = _asVector("initial value vector", initValueVector, integer=False)
        policy = _asVector("policy", policy, integer=True)
        if policy is None and not self._solved:
            raise RuntimeError("No policy has been given, and solve() has not been run on this model object, so there is no policy to evaluate. Give the policy with evaluate(policy=...), or call solve() first.")

        # The compiled module checks the length of the vectors and the actions
        # of the policy against the model (ValueError), before it changes anything
        self.mdl.evaluate(
            policy=policy,
            tolerance=tolerance,
            update=update,
            criterion=criterion,
            SORrelaxation=SORrelaxation,
            initValueVector=initValueVector,
            verbose=verbose,
            postProcessing=postProcessing,
            makeFinalCheck=makeFinalCheck,
            parallel=parallel,
        )
        self._solved = True
        self._criterion = criterion
        self._lastRun = "evaluate"
        self._warnConvergence("evaluate()", criterion, update, postProcessing)

    @_oneAtATime
    def hasConverged(self) -> bool:
        """
        Whether the last run of solve() or evaluate() converged.

        Returns:
            bool: True if the last run converged.
        """
        self._requireSolution("hasConverged")
        return bool(self.mdl.getConvergence()["converged"])

    @_oneAtATime
    def checkModel(self, tolerance: float = 1e-9) -> list:
        """
        Check the model defined by mdp() for errors, and describe each kind of
        problem found. The check reads the whole model, so it is never run by
        mdp() or solve(); call it when the results look wrong or the model may
        be malformed.

        Args:
            tolerance (float): How far the probabilities of a state-action pair may sum from 1.

        Returns:
            list: One message per kind of problem, with how many were found and the first examples. Empty if none was found.
        """
        if not self._modelDefined:
            raise RuntimeError("No model has been defined for this model object. Call mdp() before checkModel().")
        if not isinstance(tolerance, numbers.Real) or not tolerance >= 0.0:
            raise ValueError("The tolerance of checkModel() needs to be zero or a positive number.")
        return list(self.mdl.checkModel(tolerance=tolerance))

    @_oneAtATime
    def getRuntime(self) -> float:
        """
        Get the runtime of the last solver execution.

        Returns:
            float: Runtime in milliseconds.
        """
        self._requireSolution("getRuntime")
        return self.mdl.getRuntime()

    @_oneAtATime
    def getIterations(self) -> int:
        """
        Get the number of iterations of the last solver execution.
        An iteration is one policy improvement step, i.e. one update that
        chooses the best action in every state. evaluate() makes none, so
        after evaluate() it is 0.

        Returns:
            int: Number of iterations.
        """
        self._requireSolution("getIterations")
        return self.mdl.getIterations()

    @_oneAtATime
    def printPolicy(self) -> None:
        """Print the entire policy to the terminal."""
        self._requireSolution("printPolicy")
        self.mdl.printPolicy()

    @_oneAtATime
    def printValueVector(self) -> None:
        """Print the entire value vector to the terminal."""
        self._requireSolution("printValueVector")
        self.mdl.printValueVector()

    @_oneAtATime
    def getAction(self, stateIndex: int = 0) -> int:
        """
        Get the action from the optimized (or evaluated) policy for a specific state.

        Args:
            stateIndex (int): Index of the state.

        Returns:
            int: Action for the given state.

        Raises IndexError for a state outside the model.
        """
        self._requireSolution("getAction")
        return self.mdl.getAction(stateIndex=stateIndex)

    @_oneAtATime
    def getValue(self, stateIndex: int = 0) -> float:
        """
        Get the value from the optimized (or evaluated) value vector for a specific state.

        Args:
            stateIndex (int): Index of the state.

        Returns:
            float: Value for the given state.

        Raises IndexError for a state outside the model.
        """
        self._requireSolution("getValue")
        return self.mdl.getValue(stateIndex=stateIndex)

    @_oneAtATime
    def getPolicy(self) -> np.ndarray:
        """
        Get the entire optimized (or evaluated) policy.

        Returns:
            numpy.ndarray: The policy, the action index (int64) of each state.
        """
        self._requireSolution("getPolicy")
        return self.mdl.getPolicy()

    @_oneAtATime
    def getValueVector(self) -> np.ndarray:
        """
        Get the entire optimized (or evaluated) value vector.

        Returns:
            numpy.ndarray: The value vector, the value (float64) of each state.
        """
        self._requireSolution("getValueVector")
        return self.mdl.getValueVector()

    @_oneAtATime
    def getGain(self, bound: str | None = None) -> float:
        """
        Get the gain, the long-run average reward per period, from the last
        run of solve() or evaluate() under the average reward criterion.

        The smallest and largest change of the values in the last iteration
        of solve() are a lower and an upper bound on the gain: the gain of the
        optimized policy and the optimal gain both lie between them. After
        evaluate(), they bound the gain of the evaluated policy. When the run
        has converged, the bounds are less than the tolerance apart, so their
        midpoint is within half the tolerance of the gain.

        Args:
            bound (str): None for the midpoint of the bounds, 'lower' for the lower bound, or 'upper' for the upper bound (in any case).

        Returns:
            float: The gain, or the bound selected.

        Raises RuntimeError if the last run of solve() or evaluate() used the
        discounted reward criterion.
        """
        self._requireAverage("getGain", "gain")
        if not (bound is None or (isinstance(bound, str) and bound.lower() in ("lower", "upper"))):
            raise ValueError("Bound not recognized. Select either: None, 'lower', or 'upper'.")
        lower, upper = self.mdl.getGainBounds()
        if bound is None:
            return (lower + upper) / 2.0
        return lower if bound.lower() == "lower" else upper

    @_oneAtATime
    def getBias(self, stateIndex: int, referenceIndex: int = 0) -> float:
        """
        Get the bias (the relative value) of a state from the last run of
        solve() or evaluate() under the average reward criterion, with the
        bias of a reference state set to 0. Thus, the bias of a state is its
        value minus the value of the reference state.

        Args:
            stateIndex (int): Index of the state.
            referenceIndex (int): Index of the reference state, whose bias is 0.

        Returns:
            float: Bias of the given state.

        Raises RuntimeError if the last run of solve() or evaluate() used the
        discounted reward criterion, and IndexError for a state outside the
        model.
        """
        self._requireAverage("getBias", "bias")
        numberOfStates = self.mdl.getNumberOfStates()
        for name, index in (("state", stateIndex), ("reference state", referenceIndex)):
            if not isinstance(index, numbers.Integral) or isinstance(index, bool):
                raise TypeError("The %s needs to be an integer state index (got %r)." % (name, index))
            if not 0 <= index < numberOfStates:
                raise IndexError("The %s %d is outside the model, which has %d states (indices 0 to %d)." % (name, index, numberOfStates, numberOfStates - 1))
        return self.mdl.getValue(stateIndex=int(stateIndex)) - self.mdl.getValue(stateIndex=int(referenceIndex))

    def _requireAverage(self, method, quantity):
        """
        Raise RuntimeError unless the last run of solve() or evaluate() used
        the average reward criterion.
        """
        self._requireSolution(method)
        if self._criterion != "average":
            raise RuntimeError("The %s is defined under the average reward criterion, but the last run of %s() used criterion='%s'. Call %s(criterion='average') first." % (quantity, self._lastRun, self._criterion, self._lastRun))

    @_oneAtATime
    def saveToFile(self, fileName: str | None = None, type: str = "policy", filetype: str = "npy") -> None:
        """
        Save the optimized (or evaluated) policy or value vector to a file.

        Args:
            fileName (str): Name of the output file. By default result.npy, or result.csv when filetype is 'csv'.
            type (str): Type of result to save ('policy' or 'value').
            filetype (str): Type of file to save ('npy' or 'csv', in any case). 'npy' saves a NumPy .npy file of a 1D array with an entry per state, which np.load reads as the array getPolicy() (int64) or getValueVector() (float64) returns. 'csv' saves a comma-separated file with a header line and a row per state: the state index, then the action index or the value.

        A file name ending in .csv with filetype 'npy', or in .npy with
        filetype 'csv', raises a ValueError.

        Returns:
            None
        """

        self._requireSolution("saveToFile")
        if not (type == "policy" or type == "value"):
            raise ValueError("Type not recognized. Select either: 'policy' or 'value'.")
        # (in any case, e.g. "NPY")
        if not (isinstance(filetype, str) and filetype.lower() in ("npy", "csv")):
            raise ValueError("Filetype not recognized. Select either: 'npy' or 'csv'.")
        filetype = filetype.lower()
        if fileName is None:
            fileName = "result." + filetype
        # (mdp() reads a file by its extension)
        other = ".csv" if filetype == "npy" else ".npy"
        if str(fileName).lower().endswith(other):
            raise ValueError("The file name %s ends in %s, but filetype is '%s'. Use filetype='%s' to save a %s file." % (fileName, other, filetype, other[1:], "comma-separated" if filetype == "npy" else "NumPy .npy"))
        error = _openError(fileName, "w")
        if error is not None:
            # (the argument type hides the built-in type())
            raise error.__class__("The file %s could not be opened for writing (%s)." % (fileName, error.strerror or error))

        return self.mdl.saveToFile(fileName=fileName, type=type, filetype=filetype)

    @_oneAtATime
    def mdp(
        self,
        discount: float = 0.99,
        rewards: "numpy.ndarray | list" = list(),
        rewardsElementwise: "numpy.ndarray | list" = list(),
        rewardsFromFile: str = "rewards.csv",
        tranMatWithZeros: "numpy.ndarray | list" = list(),
        tranMatElementwise: "numpy.ndarray | list" = list(),
        tranMatProbs: "numpy.ndarray | list" = list(),
        tranMatColumns: "numpy.ndarray | list" = list(),
        tranMatFromFile: str = "transitions.csv",
    ) -> None:
        """
        Define the generic MDP model.

        Args:
            discount (float): Discount factor.
            rewards (NumPy array or list): The reward of each action in each state, of the shape states x actions.
            rewardsElementwise (NumPy array or list): Alternative reward format. One row per state and action: state, action, reward (n x 3).
            rewardsFromFile (str): Load the rewards from a file: a NumPy .npy file (a name ending in .npy) holding rewardsElementwise as a 2D array (of float64, float32, int64 or int32), or a comma-separated (,) file with a header line and rows state,action,reward.
            tranMatWithZeros (NumPy array or list): The transition probabilities, zeros included, of the shape states x actions x states.
            tranMatElementwise (NumPy array or list): Sparse transition probabilities (option 1). One row per non-zero probability: state, action, next state, probability (n x 4).
            tranMatProbs (NumPy array or list): Sparse transition probabilities (option 2, part 1). The non-zero transition probabilities of each state and action, of the shape states x actions x next states per pair.
            tranMatColumns (NumPy array or list): Sparse transition probabilities (option 2, part 2). The next states (columns) of the probabilities in tranMatProbs, of the same shape.
            tranMatFromFile (str): Load the transition probabilities from a file: a NumPy .npy file (a name ending in .npy) holding tranMatElementwise as a 2D array (of float64, float32, int64 or int32), or a comma-separated (,) file with a header line and rows stateFrom,action,stateTo,probability.

        Give the rewards in one format and the transition probabilities in one
        format. When neither an array nor a list is given for either, it is read
        from the file named by rewardsFromFile or tranMatFromFile, by default
        rewards.csv or transitions.csv in the working directory. The files are
        read straight into the solver's tables, so the model is not held in
        Python; .npy files are written with np.save (or, row block by row block,
        with np.lib.format.open_memmap).

        Transition probabilities given as a float32 array (tranMatWithZeros,
        tranMatElementwise or tranMatProbs), or in a float32 .npy file, are kept in single precision, which
        takes 8 instead of 12 bytes per non-zero probability; the results are
        those of the same numbers in float64. Rounded to float32, the
        probabilities of a state and action sum to 1 only to within about 1e-7,
        which can take the solver more iterations, and limits the tolerance it
        can reach under the average reward criterion.

        As arrays, the formats need the same number of actions in every state,
        and tranMatProbs and tranMatColumns the same number of next states for
        every state and action: pad with the probability 0 and any state of the
        model as the next state (e.g. 0), or use tranMatElementwise. Lists may
        have different numbers of actions and next states.

        A model object holds one model, so mdp() can be called only once per
        model object. Create a new model object to define another model.

        Returns:
            None
        """
        
        if self._modelDefined:
            raise RuntimeError("A model has already been defined for this model object, and mdp() can only be called once per model object. Create a new model object (e.g. mdl = mdpsolver.model()) to define another model.")
        if discount <= 0.0 or discount >= 1.0:
            raise ValueError("The discount needs to be in the interval between 0.0 and 1.0 (i.e. 0 < discount < 1).")

        # Each format but the files may be a NumPy array or a list. Lists are passed on as
        # they are; arrays are checked for their number of dimensions and passed on as
        # C-contiguous float64 (int64 for the column indices)
        rewards = _asInput("rewards", rewards)
        rewardsElementwise = _asInput("rewardsElementwise", rewardsElementwise)
        tranMatWithZeros = _asInput("tranMatWithZeros", tranMatWithZeros, keepFloat32=True)
        tranMatElementwise = _asInput("tranMatElementwise", tranMatElementwise, keepFloat32=True)
        tranMatProbs = _asInput("tranMatProbs", tranMatProbs, keepFloat32=True)
        tranMatColumns = _asInput("tranMatColumns", tranMatColumns, integer=True)
        # tranMatProbs and tranMatColumns go together, so an array makes both arrays
        if len(tranMatProbs) > 0 and len(tranMatColumns) > 0 and not (isinstance(tranMatProbs, list) and isinstance(tranMatColumns, list)):
            tranMatProbs = _asInput("tranMatProbs", tranMatProbs, toArray=True, keepFloat32=True)
            tranMatColumns = _asInput("tranMatColumns", tranMatColumns, integer=True, toArray=True)
            if tranMatProbs.shape != tranMatColumns.shape:
                raise ValueError("tranMatProbs and tranMatColumns as NumPy arrays need the same shape (states x actions x next states per pair), but they have the shapes %s and %s." % (tranMatProbs.shape, tranMatColumns.shape))
        
        # The formats given. A file name other than the default counts as
        # given; the default file is read only when no list is given.
        if (len(tranMatProbs) > 0) != (len(tranMatColumns) > 0):
            given, missing = ("tranMatProbs", "tranMatColumns") if len(tranMatProbs) > 0 else ("tranMatColumns", "tranMatProbs")
            raise ValueError("%s was given without %s. This format needs both: the non-zero transition probabilities in tranMatProbs and their next states in tranMatColumns." % (given, missing))
        rewardFormats = [name for name, isGiven in (
            ("rewards", len(rewards) > 0),
            ("rewardsElementwise", len(rewardsElementwise) > 0),
            ("rewardsFromFile", rewardsFromFile != "rewards.csv")) if isGiven]
        tranMatFormats = [name for name, isGiven in (
            ("tranMatWithZeros", len(tranMatWithZeros) > 0),
            ("tranMatElementwise", len(tranMatElementwise) > 0),
            ("tranMatProbs and tranMatColumns", len(tranMatProbs) > 0),
            ("tranMatFromFile", tranMatFromFile != "transitions.csv")) if isGiven]
        if len(rewardFormats) > 1:
            raise ValueError("The rewards were given in more than one format (%s). Give them in one format only." % ", ".join(rewardFormats))
        if len(tranMatFormats) > 1:
            raise ValueError("The transition probabilities were given in more than one format (%s). Give them in one format only." % ", ".join(tranMatFormats))

        # A file that cannot be opened would leave the model without rewards
        # or transition probabilities
        # (raised as the OSError that opening the file gave, e.g. FileNotFoundError)
        if rewardFormats in ([], ["rewardsFromFile"]):
            error = _openError(rewardsFromFile)
            if error is not None and rewardFormats:
                raise type(error)("The rewards file %s could not be opened (%s)." % (rewardsFromFile, error.strerror or error))
            if error is not None:
                raise type(error)("No rewards were given, and the file %s, which is read when none are given, could not be opened (%s). Give the rewards with rewards, rewardsElementwise or rewardsFromFile." % (rewardsFromFile, error.strerror or error))
        if tranMatFormats in ([], ["tranMatFromFile"]):
            error = _openError(tranMatFromFile)
            if error is not None and tranMatFormats:
                raise type(error)("The transition probabilities file %s could not be opened (%s)." % (tranMatFromFile, error.strerror or error))
            if error is not None:
                raise type(error)("No transition probabilities were given, and the file %s, which is read when none are given, could not be opened (%s). Give the transition probabilities with tranMatWithZeros, tranMatElementwise, tranMatProbs and tranMatColumns, or tranMatFromFile." % (tranMatFromFile, error.strerror or error))

        # Set before loading, so that a load that fails halfway cannot be
        # followed by another into the same object
        self._modelDefined = True
        self.mdl.mdp(
            discount=discount,
            rewards=rewards,
            rewardsElementwise=rewardsElementwise,
            rewardsFromFile=rewardsFromFile,
            tranMatWithZeros=tranMatWithZeros,
            tranMatElementwise=tranMatElementwise,
            tranMatProbs=tranMatProbs,
            tranMatColumns=tranMatColumns,
            tranMatFromFile=tranMatFromFile,
        )
