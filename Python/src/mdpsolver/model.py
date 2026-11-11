"""
MDPSolver

This file defines a `model` class that wraps around the `solvermodule` to create
and solve Markov Decision Processes (MDPs). It provides methods to configure the model,
solve it, and extract the resulting policy and value vectors.
"""

import numbers
import warnings
from mdpsolver import solvermodule


class ConvergenceWarning(UserWarning):
    """
    Issued by solve() when a run has not converged.
    """

def _openError(fileName, mode="r"):
    """The OSError raised when the file is opened in the given mode, or None if it can be opened."""
    try:
        with open(fileName, mode):
            pass
    except OSError as error:
        return error
    return None


class model:
    """
    Creates an MDPSolver object.

    This class provides methods to initialize, solve, and return results
    from an MDP model.
    """

    def __init__(self):
        """Initialize the model by creating a solver module object."""
        self.initialize()

    def initialize(self):
        """Create and initialize the solver module object."""
        self.mdl = solvermodule.Model()
        # A model object contains only one model
        self._modelDefined = False
        self._solved = False

    def _requireSolution(self, method):
        """
        Raise RuntimeError unless solve() has run on this model object.
        """
        if not self._modelDefined:
            raise RuntimeError("No model has been defined and solved for this model object, so %s() has no solution to use. Call mdp() and then solve() first." % method)
        if not self._solved:
            raise RuntimeError("The model has not been solved yet, so %s() has no solution to use. Call solve() first." % method)

    def solve(
        self,
        algorithm: str = "mpi",
        tolerance: float = 1e-3,
        update: str = "standard",
        criterion: str = "discounted",
        parIterLim: int = 100,
        SORrelaxation: float = 1.0,
        initPolicy: list = list(),
        initValueVector: list = list(),
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
            initPolicy (list): A 1D-list defining the initial policy, one valid action index per state. This option can be used for "warm starting" the optimization procedure.
            initValueVector (list): A 1D-list defining the initial value vector, one value per state. This option can be used for "warm starting" the optimization procedure.
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
        if tolerance <= 0.0 or tolerance >= 1.0:
            raise ValueError("The tolerance needs to be between 0.0 and 1.0 (i.e. 0 < tolerance < 1).")
        if not (update == "standard" or update == "gs" or update == "sor"):
            raise ValueError("Update method not recognized. Select either: 'standard', 'gs', or 'sor'.")
        if not (criterion == "discounted" or criterion == "average"):
            raise ValueError("Optimality criterion not recognized. Select either: 'discounted' or 'average'.")
        if criterion == "average" and (update == "gs" or update == "sor"):
            raise ValueError("Gauss-Seidel and SOR updates are not available with the average reward criterion. Select update='standard'.")
        if parIterLim <= 0 and algorithm == "mpi":
            raise ValueError("The partial evaluation limit needs to be a positive number.")
        if (SORrelaxation <= 0.0 or SORrelaxation >= 2.0) and update == "sor":
            raise ValueError("The relaxation parameter needs to be between 0.0 and 2.0 (i.e. 0 < SORrelaxation < 2).")
        if len(initPolicy) > 0 or len(initValueVector) > 0:
            numberOfStates = self.mdl.getNumberOfStates()
            if numberOfStates >= 0 and len(initPolicy) > 0 and len(initPolicy) != numberOfStates:
                raise ValueError("The initial policy needs one action index per state (the model has %d states, initPolicy has %d elements)." % (numberOfStates, len(initPolicy)))
            if numberOfStates >= 0 and len(initValueVector) > 0 and len(initValueVector) != numberOfStates:
                raise ValueError("The initial value vector needs one value per state (the model has %d states, initValueVector has %d elements)." % (numberOfStates, len(initValueVector)))
            for stateIndex, value in enumerate(initValueVector):
                if not isinstance(value, numbers.Real):
                    raise TypeError("The initial value vector needs a number for every state (state %d has %r)." % (stateIndex, value))
            if numberOfStates >= 0 and len(initPolicy) > 0:
                for stateIndex, action in enumerate(initPolicy):
                    if not isinstance(action, numbers.Integral):
                        raise TypeError("The initial policy needs an integer action index for every state (state %d has %r)." % (stateIndex, action))
                    numberOfActions = self.mdl.getNumberOfActions(stateIndex=stateIndex)
                    if not 0 <= action < numberOfActions:
                        raise ValueError("The initial policy assigns action %s to state %d, which has %d actions (indices 0 to %d)." % (action, stateIndex, numberOfActions, numberOfActions - 1))

        # The solver keeps the policy and the value vector of the previous run,
        # unless initPolicy or initValueVector replace them.

        if self._solved:
            # Value iteration never uses the policy it starts from
            reusesPolicy = len(initPolicy) == 0 and algorithm != "vi"
            reusesValueVector = len(initValueVector) == 0
            if reusesPolicy or reusesValueVector:
                print("Note: This model has been solved before. Create a new model object to run solver with initial values.", flush=True)

        self.mdl.solve(
            algorithm=algorithm,
            tolerance=tolerance,
            update=update,
            criterion=criterion,
            parIterLim=parIterLim,
            SORrelaxation=SORrelaxation,
            initPolicy=initPolicy,
            initValueVector=initValueVector,
            verbose=verbose,
            postProcessing=postProcessing,
            makeFinalCheck=makeFinalCheck,
            parallel=parallel,
        )
        # Set after the run, so that a run that did not finish leaves the
        # object without a solution
        self._solved = True

        convergence = self.mdl.getConvergence()
        hint = " Under the average reward criterion, this happens with periodic models." if criterion == "average" else ""
        if convergence["iterationLimitReached"]:
            warnings.warn("solve() stopped at the iteration limit of %d iterations before reaching the tolerance, so the results have not converged.%s" % (convergence["iterationLimit"], hint), ConvergenceWarning, stacklevel=2)
        elif convergence["evaluationLimitReached"]:
            warnings.warn("A policy evaluation of policy iteration stopped at its limit of %d iterations before reaching the tolerance, so the results have not converged.%s" % (convergence["iterationLimit"], hint), ConvergenceWarning, stacklevel=2)
        if convergence["valuesOutOfRange"] and not postProcessing and update == "standard":
            # The values are left uncorrected, and may lie outside the range
            # even when the policy is right
            warnings.warn("The final check found the value %r in state %d, outside the range that the rewards and the discount allow. With postProcessing=False, the values are not corrected and may lie outside it; use postProcessing=True for correct values, or makeFinalCheck=False to skip the check." % (convergence["outOfRangeValue"], convergence["outOfRangeState"]), ConvergenceWarning, stacklevel=2)
        elif convergence["valuesOutOfRange"]:
            warnings.warn("The final check found the value %r in state %d, outside the range that the rewards and the discount allow, so the results are wrong. checkModel() can look for errors in the model." % (convergence["outOfRangeValue"], convergence["outOfRangeState"]), ConvergenceWarning, stacklevel=2)
        elif convergence["diverged"]:
            warnings.warn("solve() stopped because the values became infinite or NaN, so the results are wrong. checkModel() can look for errors in the model.", ConvergenceWarning, stacklevel=2)

    def hasConverged(self) -> bool:
        """
        Whether the last run of solve() converged.

        Returns:
            bool: True if the last run converged.
        """
        self._requireSolution("hasConverged")
        return bool(self.mdl.getConvergence()["converged"])

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

    def getRuntime(self) -> float:
        """
        Get the runtime of the last solver execution.

        Returns:
            float: Runtime in milliseconds.
        """
        self._requireSolution("getRuntime")
        return self.mdl.getRuntime()

    def getIterations(self) -> int:
        """
        Get the number of iterations of the last solver execution.
        An iteration is one policy improvement step, i.e. one update that
        chooses the best action in every state.

        Returns:
            int: Number of iterations.
        """
        self._requireSolution("getIterations")
        return self.mdl.getIterations()

    def printPolicy(self) -> None:
        """Print the entire policy to the terminal."""
        self._requireSolution("printPolicy")
        self.mdl.printPolicy()

    def printValueVector(self) -> None:
        """Print the entire value vector to the terminal."""
        self._requireSolution("printValueVector")
        self.mdl.printValueVector()

    def getAction(self, stateIndex: int = 0) -> int:
        """
        Get the action from the optimized policy for a specific state.

        Args:
            stateIndex (int): Index of the state.

        Returns:
            int: Action for the given state.

        Raises IndexError for a state outside the model.
        """
        self._requireSolution("getAction")
        return self.mdl.getAction(stateIndex=stateIndex)

    def getValue(self, stateIndex: int = 0) -> float:
        """
        Get the value from the optimized value vector for a specific state.

        Args:
            stateIndex (int): Index of the state.

        Returns:
            float: Value for the given state.

        Raises IndexError for a state outside the model.
        """
        self._requireSolution("getValue")
        return self.mdl.getValue(stateIndex=stateIndex)

    def getPolicy(self) -> list:
        """
        Get the entire optimized policy.

        Returns:
            list: Optimized policy.
        """
        self._requireSolution("getPolicy")
        return self.mdl.getPolicy()

    def getValueVector(self):
        """
        Get the entire optimized value vector.

        Returns:
            list: Optimized value vector.
        """
        self._requireSolution("getValueVector")
        return self.mdl.getValueVector()

    def saveToFile(self, fileName: str = "result.csv", type: str = "policy") -> None:
        """
        Save the optimized policy or value vector to a file.

        Args:
            fileName (str): Name of the output file.
            type (str): Type of result to save ('policy' or 'value').

        Returns:
            None
        """
        
        self._requireSolution("saveToFile")
        if not (type == "policy" or type == "value"):
            raise ValueError("Type not recognized. Select either: 'policy' or 'value'.")
        error = _openError(fileName, "w")
        if error is not None:
            # (the argument type hides the built-in type())
            raise error.__class__("The file %s could not be opened for writing (%s)." % (fileName, error.strerror or error))

        return self.mdl.saveToFile(fileName=fileName, type=type)

    def mdp(
        self,
        discount: float = 0.99,
        rewards: list = list(),
        rewardsElementwise: list = list(),
        rewardsFromFile: str = "rewards.csv",
        tranMatWithZeros: list = list(),
        tranMatElementwise: list = list(),
        tranMatProbs: list = list(),
        tranMatColumns: list = list(),
        tranMatFromFile: str = "transitions.csv",
    ) -> None:
        """
        Define the generic MDP model.

        Args:
            discount (float): Discount factor.
            rewards (list): A 2D-list containing the reward (float) of a particular action in a particular state.
            rewardsElementwise (list): Alternative reward format. A 2D-list where each row corresponds to a combination of a state and an action.
            rewardsFromFile (str): Load the rewards from a comma-separated (,) file.
            tranMatWithZeros (list): A 3D-list containing the transition probabilities.
            tranMatElementwise (list): Sparse transition probabilities (option 1). A 2D-list where each row corresponds to a combination of a current state, an action, and a next state.
            tranMatProbs (list): Sparse transition probabilities (option 2, part 1). A 3D-list containing the non-zero transition probabilities.
            tranMatColumns (list): Sparse transition probabilities (option 2, part 2). A 3D-list containing the columns of the non-zero transition probabilities.
            tranMatFromFile (str): Load the transition probabilities from a comma-separated (,) file.

        Give the rewards in one format and the transition probabilities in one
        format. When no list is given for either, it is read from the file
        named by rewardsFromFile or tranMatFromFile, by default rewards.csv or
        transitions.csv in the working directory.

        A model object holds one model, so mdp() can be called only once per
        model object. Create a new model object to define another model.

        Returns:
            None
        """
        
        if self._modelDefined:
            raise RuntimeError("A model has already been defined for this model object, and mdp() can only be called once per model object. Create a new model object (e.g. mdl = mdpsolver.model()) to define another model.")
        if discount <= 0.0 or discount >= 1.0:
            raise ValueError("The discount needs to be in the interval between 0.0 and 1.0 (i.e. 0 < discount < 1).")
        
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
