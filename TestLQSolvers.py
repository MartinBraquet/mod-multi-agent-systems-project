import numpy as np
from LQFeedbackSolver import * 
from Utils import * 
from copy import deepcopy
# Unit tests for LQ Nash solvers.

# Common dynamics, costs, and initial condition.
A = np.array([[1, 0.1, 0, 0  ],
             [0, 1,   0, 0  ],
             [0, 0,   1, 1.1],
             [0, 0,   0, 1  ]])
B1 = np.array([0, 0.1, 0, 0], ndmin = 2).T
B2 = np.array([0, 0  , 0, 0.1], ndmin = 2).T
dyn = Dynamics(A, [B1, B2])

Q1 = np.array([[0, 0, 0,  0],
              [0, 0, 0,  0],
              [0, 0, 1., 0],
              [0, 0, 0,  0]])
c1 = Cost(Q1)
R1 = np.ones((1, 1))
add_control_cost(c1, 0, R1)

Q2 = np.array([[1., 0, -1, 0],
              [0,  0, 0,  0],
              [-1, 0, 1,  0],
              [0,  0, 0,  0]])
c2 = Cost(Q2)
R2 = np.ones((1, 1))
add_control_cost(c2, 1, R2)

costs = [c1, c2]

x1 = np.array([1., 0, 1, 0])
horizon = 10

# Ensure that the feedback solution satisfies Nash conditions of optimality
# for each player, holding others' strategies fixed.
Ps = solve_lq_feedback(dyn, costs, horizon)
xs, us = unroll_feedback(dyn, Ps, x1)
nash_costs = [evaluate(c, xs, us) for c in costs]

        # Perturb each strategy a little bit and confirm that cost only
        # increases for that player.
eps = 1e-1
for ii in range(2):
    for tt in range(horizon):
        P̃s = deepcopy(Ps)
        P̃s[ii][:, :, tt] += eps * np.random.random((udim(dyn, ii), xdim(dyn)))
     
        x̃s, ũs = unroll_feedback(dyn, P̃s, x1)
        new_nash_costs = [evaluate(c, x̃s, ũs) for c in costs]
        assert new_nash_costs[ii] >= nash_costs[ii]
