"""
RUNTIME TEST: VALUE ITERATION, MDPSOLVER AGAINST PYMDPTOOLBOX

The models, settings and measurements are described in runtime_tests_common.py.
Results: results/mdp_test_results_vi.csv.
"""

import mdptoolbox.mdp

from runtime_tests_common import DISCOUNT, runTests

tolerance = 1e-3

runTests("vi",
         dict(algorithm="vi", update="standard", tolerance=tolerance),
         lambda transitions, rewards: mdptoolbox.mdp.ValueIteration(
             transitions, rewards, DISCOUNT, epsilon=tolerance))
