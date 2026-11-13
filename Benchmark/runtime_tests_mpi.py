"""
RUNTIME TEST: MODIFIED POLICY ITERATION, MDPSOLVER AGAINST PYMDPTOOLBOX

The models, settings and measurements are described in runtime_tests_common.py.
Results: results/mdp_test_results_mpi.csv.
"""

import mdptoolbox.mdp

from runtime_tests_common import DISCOUNT, runTests

tolerance = 1e-3
parIterLim = 100  # the limit of the partial policy evaluation

runTests("mpi",
         dict(algorithm="mpi", update="standard", tolerance=tolerance, parIterLim=parIterLim),
         lambda transitions, rewards: mdptoolbox.mdp.PolicyIterationModified(
             transitions, rewards, DISCOUNT, epsilon=tolerance, max_iter=parIterLim))
