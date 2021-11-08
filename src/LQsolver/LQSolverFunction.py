import numpy as np
from LQFeedbackSolver import * 
from Utils import * 

def LQFeedbackFunction(A, B1, B2, Q1, Q2, R1, R2, x1, horizon):

    dyn = Dynamics(A, [B1, B2])
    
    c1 = Cost(Q1)
    add_control_cost(c1, 0, R1)
    
    c2 = Cost(Q2)
    add_control_cost(c2, 1, R2)
    
    costs = [c1, c2]
    
    # Ensure that the feedback solution satisfies Nash conditions of optimality
    # for each player, holding others' strategies fixed.
    Ps = solve_lq_feedback(dyn, costs, horizon)
    xs, us = unroll_feedback(dyn, Ps, x1)
    nash_costs = [evaluate(c, xs, us) for c in costs]
    
    return xs, us, nash_costs



A = np.array([[1, 0.1, 0, 0  ],
             [0, 1,   0, 0  ],
             [0, 0,   1, 1.1],
             [0, 0,   0, 1  ]])
B1 = np.array([0, 0.1, 0, 0], ndmin = 2).T
B2 = np.array([0, 0  , 0, 0.1], ndmin = 2).T

Q1 = np.array([[0, 0, 0,  0],
              [0, 0, 0,  0],
              [0, 0, 1., 0],
              [0, 0, 0,  0]])
R1 = np.ones((1, 1))

Q2 = np.array([[1., 0, -1, 0],
              [0,  0, 0,  0],
              [-1, 0, 1,  0],
              [0,  0, 0,  0]])
R2 = np.ones((1, 1))

x1 = np.array([1., 0, 1, 0])
horizon = 10

xs, us, nash_costs = LQFeedbackFunction(A, B1, B2, Q1, Q2, R1, R2, x1, horizon)