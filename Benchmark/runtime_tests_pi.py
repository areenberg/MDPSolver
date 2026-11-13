"""
RUNTIME TEST: POLICY ITERATION, MDPSOLVER AGAINST PYMDPTOOLBOX

The models, settings and measurements are described in runtime_tests_common.py.
Results: results/mdp_test_results_pi.csv.
"""

import mdptoolbox.mdp

from runtime_tests_common import DISCOUNT, runTests

tolerance = 1e-4

runTests("pi",
         dict(algorithm="pi", update="standard", tolerance=tolerance),
         lambda transitions, rewards: mdptoolbox.mdp.PolicyIteration(
             transitions, rewards, DISCOUNT))
