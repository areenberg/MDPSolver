'''
PROBLEMS WITH KNOWN SOLUTIONS, AND EXACT SOLUTIONS, FOR THE TESTS IN THIS FOLDER.

A problem is given as rewards[s][a] and transitions[s][a] = {next state:
probability}, from which the tests build the arguments of mdp().

The hub-and-spoke problem. In the hub (state 0) one either rests, for a reward
REST per period, or starts one of three ventures, which succeeds with
probability 1 - FAIL. Venture k (state k) pays R_k per period and ends with
probability q_k per period, back in the hub; two of the ventures can also be
quit, for nothing, back to the hub. All rewards are non-negative. With discount
factor lambda, the hub is worth m/(1 - lambda), where m is the best of REST and
lambda(1 - FAIL)R_k/(1 + lambda(q_k - FAIL)), the per-period worth of resting or
of venture k; a venture is continued if R_k >= lambda(1 - q_k)m, and is then
worth (R_k + lambda q_k v_hub)/(1 - lambda(1 - q_k)), and is quit otherwise,
worth lambda v_hub. The constants are chosen so that the best choice in the hub
moves from resting, to the short venture, to the long one as lambda grows, and
the poor venture goes from continued to quit. Under the average-reward criterion
the gain is the best of REST and (1 - FAIL)R_k/(1 + q_k - FAIL) (a renewal
argument), and with the bias of the hub set to 0, a continued venture has bias
(R_k - g)/q_k and a quit one -g.

The secretary problem (Ferguson, 1989). NCANDIDATES candidates are seen one at
a time in random order, and each can be selected, ending the search, or passed
over for good; the aim is to select the best of all. The optimal rule passes
over the first CUTOFF - 1 candidates and then selects the first one that is
better than all before it, which wins with probability about 1/e (0.398690 for
ten candidates). The states are a candidate's position i and whether it is the
best so far, plus an absorbing state once a candidate is selected or none is
left. The classical problem is undiscounted and MDPSolver discounts every step,
but every candidate takes exactly one step, so a reward of (i/n)/lambda^(i - 1)
for selecting candidate i undoes the discount exactly: the start state is worth
the classical probability of winning at every lambda, and candidate i's states
are worth their classical values divided by lambda^(i - 1).

The exact solutions of any problem: exactdiscounted() and exactaverage() run
policy iteration with every policy evaluated exactly, by solving its linear
equations, so they share no arithmetic with MDPSolver's iterative evaluation.
checkoptimality() checks a solution against the optimality equation, with a
margin for the optimal actions, and unichainaperiodic() the condition under
which value iteration converges under the average-reward criterion (Puterman,
1994, Section 8.5).

References:
    Ferguson, T. S. (1989). Who solved the secretary problem? Statistical
    Science 4(3), 282-289. https://doi.org/10.1214/ss/1177012493
    Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic
    Dynamic Programming. Wiley. https://doi.org/10.1002/9780470316887
'''

import itertools
import sys

import numpy as np


# ------------------------------------------------------------------
# THE HUB-AND-SPOKE PROBLEM
# ------------------------------------------------------------------

# The reward for resting in the hub, the probability that starting a venture
# fails, and the ventures as (reward per period, probability of ending per
# period, whether it can be quit): a short one, a long one and a poor one.
REST = 2.0
FAIL = 0.1
VENTURES = ((6.0, 0.6, False),
            (4.2, 0.05, True),
            (1.5, 0.3, True))


def hubproblem():
    '''The rewards and transitions of the hub-and-spoke problem.'''
    nventures = len(VENTURES)
    rewards = [[REST] + [0.0] * nventures]
    transitions = [[{0: 1.0}] + [{0: FAIL, k: 1.0 - FAIL} for k in range(1, nventures + 1)]]
    for k, (reward, end, quittable) in enumerate(VENTURES, start=1):
        rewards.append([reward] + ([0.0] if quittable else []))
        transitions.append([{0: end, k: 1.0 - end}] + ([{0: 1.0}] if quittable else []))
    return rewards, transitions


def hubdiscounted(discount):
    '''The optimal policy and values of the hub-and-spoke problem under discounting.'''
    worths = [REST] + [discount * (1.0 - FAIL) * reward / (1.0 + discount * (end - FAIL))
                       for reward, end, _ in VENTURES]
    best = max(worths)
    hub = best / (1.0 - discount)
    policy, values = [int(np.argmax(worths))], [hub]
    for reward, end, quittable in VENTURES:
        if quittable and reward < discount * (1.0 - end) * best:
            policy.append(1)
            values.append(discount * hub)
        else:
            policy.append(0)
            values.append((reward + discount * end * hub) / (1.0 - discount * (1.0 - end)))
    return policy, np.array(values)


def hubaverage():
    '''The gain-optimal policy, the gain and the bias (0 in the hub) of the hub-and-spoke problem.'''
    gains = [REST] + [(1.0 - FAIL) * reward / (1.0 + end - FAIL) for reward, end, _ in VENTURES]
    gain = max(gains)
    policy, bias = [int(np.argmax(gains))], [0.0]
    for reward, end, quittable in VENTURES:
        if quittable and reward < gain * (1.0 - end):
            policy.append(1)
            bias.append(-gain)
        else:
            policy.append(0)
            bias.append((reward - gain) / end)
    return policy, gain, np.array(bias)


# ------------------------------------------------------------------
# THE SECRETARY PROBLEM
# ------------------------------------------------------------------

NCANDIDATES = 10
WINPROBABILITY = 0.398690   # for ten candidates (Ferguson, 1989), to six decimals


def secretaryproblem(discount, compensated=True):
    '''
    The rewards and transitions of the secretary problem. State 2(i - 1) is
    candidate i being the best so far, state 2(i - 1) + 1 it not being so, and
    state 2n the end. A best-so-far candidate can be passed over (action 0) or
    selected (action 1); any other is passed over. Compensated, the reward for
    selecting candidate i undoes the discount of the i - 1 steps to it;
    otherwise it is i/n whatever the discount.
    '''
    n = NCANDIDATES
    end = 2 * n
    rewards, transitions = [], []
    for i in range(1, n + 1):
        onward = {end: 1.0} if i == n else {2 * i: 1.0 / (i + 1), 2 * i + 1: i / (i + 1)}
        rewards.append([0.0, (i / n) / (discount ** (i - 1) if compensated else 1.0)])
        transitions.append([onward, {end: 1.0}])
        rewards.append([0.0])
        transitions.append([dict(onward)])
    rewards.append([0.0])
    transitions.append([{end: 1.0}])
    return rewards, transitions


def secretarysolution(discount):
    '''
    The optimal policy and values of the compensated secretary problem, from
    the classical solution: W(m) = (m/n) sum_{k=m}^{n-1} 1/k is the probability
    of winning by selecting the next best-so-far candidate after the m-th, and
    the cutoff is the first candidate from which selecting a best-so-far one is
    at least as good as going on. Returns the cutoff too.
    '''
    n = NCANDIDATES

    def W(m):
        return (m / n) * sum(1.0 / k for k in range(m, n))

    cutoff = min(r for r in range(1, n + 1) if sum(1.0 / k for k in range(r, n)) <= 1.0)
    policy, values = [], []
    for i in range(1, n + 1):
        onward = W(max(i, cutoff - 1))
        scale = discount ** (i - 1)
        policy += [1 if i >= cutoff else 0, 0]
        values += [max(i / n, onward) / scale, onward / scale]
    policy.append(0)
    values.append(0.0)
    return cutoff, policy, np.array(values)


# ------------------------------------------------------------------
# ANY PROBLEM
# ------------------------------------------------------------------

def densetranmat(transitions):
    '''The transition probabilities as a 3D list with zeros, as tranMatWithZeros.'''
    n = len(transitions)
    return [[[probs.get(j, 0.0) for j in range(n)] for probs in state] for state in transitions]


def elementwisetranmat(transitions):
    '''The non-zero transition probabilities as rows [s, a, j, probability], as tranMatElementwise.'''
    return [[s, a, j, p] for s, state in enumerate(transitions)
            for a, probs in enumerate(state) for j, p in probs.items()]


def matrices(rewards, transitions):
    '''Per state, the rewards of its actions and the rows of their transition probabilities.'''
    n = len(rewards)
    R = [np.array(state, dtype=float) for state in rewards]
    P = []
    for state in transitions:
        rows = np.zeros((len(state), n))
        for a, probs in enumerate(state):
            for j, p in probs.items():
                rows[a, j] = p
        P.append(rows)
    return R, P


def improve(R, P, policy, values, discount):
    '''The greedy policy for the values, keeping an action that ties with the best.'''
    improved = []
    for s in range(len(R)):
        q = R[s] + discount * (P[s] @ values)
        best = int(np.argmax(q))
        tie = q[policy[s]] >= q[best] - 1e-12 * max(1.0, abs(q[best]))
        improved.append(policy[s] if tie else best)
    return improved


def exactdiscounted(rewards, transitions, discount):
    '''The optimal policy and values under discounting, by policy iteration with exact evaluation.'''
    R, P = matrices(rewards, transitions)
    n = len(R)
    policy = [int(np.argmax(r)) for r in R]
    while True:
        Ppolicy = np.array([P[s][policy[s]] for s in range(n)])
        rpolicy = np.array([R[s][policy[s]] for s in range(n)])
        values = np.linalg.solve(np.eye(n) - discount * Ppolicy, rpolicy)
        improved = improve(R, P, policy, values, discount)
        if improved == policy:
            return policy, values
        policy = improved


def exactaverage(rewards, transitions):
    '''
    The gain-optimal policy, the gain and the bias (0 in state 0) of a unichain
    problem, by policy iteration with exact evaluation: h + g = r + P h, h(0) = 0.
    '''
    R, P = matrices(rewards, transitions)
    n = len(R)
    policy = [int(np.argmax(r)) for r in R]
    while True:
        Ppolicy = np.array([P[s][policy[s]] for s in range(n)])
        rpolicy = np.array([R[s][policy[s]] for s in range(n)])
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = np.eye(n) - Ppolicy
        A[:n, n] = 1.0
        A[n, 0] = 1.0
        solution = np.linalg.solve(A, np.r_[rpolicy, 0.0])
        bias, gain = solution[:n], solution[n]
        improved = improve(R, P, policy, bias, 1.0)
        if improved == policy:
            return policy, gain, bias
        policy = improved


def qvalues(rewards, transitions, values, discount):
    '''r(s, a) + discount * sum_j p(j | s, a) v(j) for every state and action.'''
    R, P = matrices(rewards, transitions)
    return [list(R[s] + discount * (P[s] @ np.asarray(values))) for s in range(len(R))]


def checkoptimality(name, rewards, transitions, policy, values, discount, mingap, gain=0.0):
    '''
    Ends the test unless the values solve the optimality equation
    v + g = max_a (r + discount P v), with the policy as the maximizer and every
    maximum ahead of the runner-up by mingap.
    '''
    q = qvalues(rewards, transitions, values, discount)
    scale = max(1.0, float(np.max(np.abs(values))))
    for s in range(len(rewards)):
        if abs(values[s] + gain - max(q[s])) > 1e-11 * scale:
            sys.exit("Validation test failed at initialization: the known solution of the %s does not solve the optimality equation in state %d." % (name, s))
        if int(np.argmax(q[s])) != policy[s]:
            sys.exit("Validation test failed at initialization: the known policy of the %s does not maximize in state %d." % (name, s))
        if len(q[s]) > 1 and sorted(q[s])[-1] - sorted(q[s])[-2] < mingap:
            sys.exit("Validation test failed at initialization: in the %s the best action of state %d wins by only %.3e, so the comparison of policies would hinge on rounding." % (name, s, sorted(q[s])[-1] - sorted(q[s])[-2]))


def smallestgap(rewards, transitions, values, discount):
    '''The smallest margin by which the best action of a state beats its runner-up.'''
    q = qvalues(rewards, transitions, values, discount)
    gaps = [sorted(qs)[-1] - sorted(qs)[-2] for qs in q if len(qs) > 1]
    return min(gaps) if gaps else np.inf


def unichainaperiodic(rewards, transitions):
    '''The first policy that is not unichain and aperiodic, or None. Enumerates every policy.'''
    n = len(rewards)
    for policy in itertools.product(*[range(len(r)) for r in rewards]):
        step = np.zeros((n, n), dtype=np.int64)
        for s in range(n):
            for j, p in transitions[s][policy[s]].items():
                step[s, j] = p > 0.0
        reach = np.eye(n, dtype=np.int64)
        for _ in range(n):
            reach = np.minimum(reach + reach @ step, 1)
        recurrent = [s for s in range(n) if all(reach[j, s] for j in range(n) if reach[s, j])]
        classes = {frozenset(j for j in recurrent if reach[s, j] and reach[j, s]) for s in recurrent}
        if len(classes) != 1:
            return policy
        members = sorted(next(iter(classes)))
        power = sub = step[np.ix_(members, members)]
        for _ in range(len(members) ** 2):   # primitive: some power is positive
            power = np.minimum(power @ sub, 1)
        if not np.all(power > 0):
            return policy
    return None
