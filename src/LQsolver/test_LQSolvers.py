import numpy as np
from LQFeedbackSolverUnroll import *
from Utils import *
from copy import deepcopy
# Unit tests for LQ Nash solvers.

horizon = 10

# Common dynamics, costs, and initial condition.
A = np.array([[1, 0.1, 0, 0  ],
             [0, 1,   0, 0  ],
             [0, 0,   1, 0.1],
             [0, 0,   0, 1  ]])
B1 = np.array([0, 0.1, 0, 0], ndmin = 2).T
B2 = np.array([0, 0  , 0, 0.1], ndmin = 2).T

mu0 = np.array([1., 0, 1, 0])
sigma0 = np.diag((.001, .001, .01, .01))
sigmaW = np.array([1e-3, 1e-3, 1e-3, 1e-3])

dyn = Dynamics(A, [B1, B2], mu0, sigma0, sigmaW)

Q1 = np.array([[0, 0, 0,  0],
              [0, 0, 0,  0],
              [0, 0, 1., 0],
              [0, 0, 0,  0]])
Q1list = [Q1] * horizon
q1 = np.array([1, 0, 0, 0]).T
q1list = [q1] * horizon
c1 = Cost(Q1list, q1list)
R1 = np.ones((1, 1))
add_control_cost(c1, 0, R1)

Q2 = np.array([[1., 0, -1, 0],
              [0,  0, 0,  0],
              [-1, 0, 1,  0],
              [0,  0, 0,  0]])
Q2list = [Q2] * horizon
q2 = np.array([0, 0, 0, 0]).T
q2list = [q2] * horizon
c2 = Cost(Q2list, q2list)
R2 = np.ones((1, 1))
add_control_cost(c2, 1, R2)

costs = [c1, c2]


# Ensure that the feedback solution satisfies Nash conditions of optimality
# for each player, holding others' strategies fixed.
Ps, a = solve_lq_feedback(dyn, costs, horizon)
xs, sigmas, us = unroll_feedback(dyn, Ps, a)
nash_costs = [evaluate(c, xs, us) for c in costs]

# Perturb each strategy a little bit and confirm that cost only
# increases for that player.
eps = 1e-1
for ii in range(2):
    for tt in range(horizon):
        Psp = deepcopy(Ps)
        Psp[ii][:, :, tt] += eps * np.random.random((udim(dyn, ii), xdim(dyn)))

        xsp, sigmasp, usp = unroll_feedback(dyn, Psp, a)
        new_nash_costs = [evaluate(c, xsp, usp) for c in costs]
        assert new_nash_costs[ii] >= nash_costs[ii]


        asp = deepcopy(a)
        asp[ii][ :, tt] += eps * np.random.random(udim(dyn, ii))

        xsp, sigmasp, usp = unroll_feedback(dyn, Ps, asp)
        new_nash_costs = [evaluate(c, xsp, usp) for c in costs]
        assert new_nash_costs[ii] >= nash_costs[ii]
