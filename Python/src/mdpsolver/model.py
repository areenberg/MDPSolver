"""
MDPSolver

This file defines a `model` class that wraps around the `solvermodule` to create
and solve Markov Decision Processes (MDPs). It provides methods to configure the model,
solve it, and extract the resulting policy and value vectors.
"""

import sys
from mdpsolver import solvermodule

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
            tolerance (float): Convergence threshold for the algorithm.
            update (str): The value-update method.
            criterion (str): The optimality criterion.
            parIterLim (int): The partial evaluation limit employed in the modified policy iteration algorithm.
            SORrelaxation (float): Relaxation parameter for the Successive Over-Relaxation method.
            initPolicy (list): A 1D-list defining the initial policy. This option can be used for "warm starting" the optimization procedure.
            initValueVector (list): A 1D-list defining the initial value vector. This option can be used for "warm starting" the optimization procedure.
            verbose (bool): If True, prints solver progress to console.
            postProcessing (bool): If True, performs post-processing after solving.
            makeFinalCheck (bool): If True, makes a final check of the value vector. This process checks if the resulting values are reasonable.
            parallel (bool): If True, enables parallel computation for faster solving. This option is only available for the custom MDP model with standard updates.

        Returns:
            None
        """
        
        if not (algorithm == "mpi" or algorithm == "pi" or algorithm == "vi"):
            sys.exit("Error: Algorithm type not recognized. Select either: 'mpi', 'pi', or 'vi'.")
        if tolerance <= 0.0:
            sys.exit("Error: The tolerance needs to be a positive number.")
        if not (update == "standard" or update == "gs" or update == "sor"):
            sys.exit("Error: Update method not recognized. Select either: 'standard', 'gs', or 'sor'.")
        if not (criterion == "discounted" or criterion == "average"):
            sys.exit("Error: Optimality criterion not recognized. Select either: 'discounted' or 'average'.")
        if parIterLim <= 0 and algorithm == "mpi":
            sys.exit("Error: The partial evaluation limit needs to be a positive number.")
        if (SORrelaxation <= 0.0 or SORrelaxation >= 2.0) and update == "sor":
            sys.exit("Error: The relaxation parameter needs to be between 0.0 and 2.0 (i.e. 0 < SORrelaxation < 2).")
        
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

    def getRuntime(self) -> float:
        """
        Get the runtime of the last solver execution.

        Returns:
            float: Runtime in milliseconds.
        """
        return self.mdl.getRuntime()

    def printPolicy(self) -> None:
        """Print the entire policy to the terminal."""
        self.mdl.printPolicy()

    def printValueVector(self) -> None:
        """Print the entire value vector to the terminal."""
        self.mdl.printValueVector()

    def getAction(self, stateIndex: int = 0) -> int:
        """
        Get the action from the optimized policy for a specific state.

        Args:
            stateIndex (int): Index of the state.

        Returns:
            int: Action for the given state.
        """
        return self.mdl.getAction(stateIndex=stateIndex)

    def getValue(self, stateIndex: int = 0) -> float:
        """
        Get the value from the optimized value vector for a specific state.

        Args:
            stateIndex (int): Index of the state.

        Returns:
            float: Value for the given state.
        """
        return self.mdl.getValue(stateIndex=stateIndex)

    def getPolicy(self) -> list:
        """
        Get the entire optimized policy.

        Returns:
            list: Optimized policy.
        """
        return self.mdl.getPolicy()

    def getValueVector(self):
        """
        Get the entire optimized value vector.

        Returns:
            list: Optimized value vector.
        """
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
        
        if not (type == "policy" or type == "value"):
            sys.exit("Error: Type not recognized. Select either: 'policy' or 'value'.")

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

        Returns:
            None
        """
        
        if discount <= 0.0 or discount >= 1.0:
            sys.exit("Error: The discount needs to be in the interval between 0.0 and 1.0 (i.e. 0 < discount < 1).")
        
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
