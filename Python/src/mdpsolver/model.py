"""
MDPSolver

This file defines a `model` class that wraps around the `solvermodule` to create
and solve Markov Decision Processes (MDPs). It provides methods to configure the model,
solve it, and extract the resulting policy and value vectors.
"""

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
        algorithm="mpi",
        tolerance=1e-3,
        update="standard",
        criterion="discounted",
        parIterLim=100,
        SORrelaxation=1.0,
        initPolicy=list(),
        initValueVector=list(),
        verbose=False,
        postProcessing=True,
        makeFinalCheck=True,
        parallel=True,
    ):
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

    def getRuntime(self):
        """
        Get the runtime of the last solver execution.

        Returns:
            float: Runtime in milliseconds.
        """
        return self.mdl.getRuntime()

    def printPolicy(self):
        """Print the entire policy to the terminal."""
        self.mdl.printPolicy()

    def printValueVector(self):
        """Print the entire value vector to the terminal."""
        self.mdl.printValueVector()

    def getAction(self, stateIndex=0):
        """
        Get the action from the optimized policy for a specific state.

        Args:
            stateIndex (int): Index of the state.

        Returns:
            int: Action for the given state.
        """
        return self.mdl.getAction(stateIndex=stateIndex)

    def getValue(self, stateIndex=0):
        """
        Get the value from the optimized value vector for a specific state.

        Args:
            stateIndex (int): Index of the state.

        Returns:
            float: Value for the given state.
        """
        return self.mdl.getValue(stateIndex=stateIndex)

    def getPolicy(self):
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

    def saveToFile(self, fileName="result.csv", type="policy"):
        """
        Save the optimized policy or value vector to a file.

        Args:
            fileName (str): Name of the output file.
            type (str): Type of result to save ('policy' or 'value').

        Returns:
            None
        """
        return self.mdl.saveToFile(fileName=fileName, type=type)

    def mdp(
        self,
        discount=0.99,
        rewards=list(),
        rewardsElementwise=list(),
        rewardsFromFile="rewards.csv",
        tranMatWithZeros=list(),
        tranMatElementwise=list(),
        tranMatProbs=list(),
        tranMatColumns=list(),
        tranMatFromFile="transitions.csv",
    ):
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
