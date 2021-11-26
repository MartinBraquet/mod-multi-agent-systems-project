import numpy as np
from LQFeedbackSolverUnroll import *
from Utils import *

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
q1 = np.array([1, 0, 0, 0]).T
R1 = np.ones((1, 1))

Q2 = np.array([[1., 0, -1, 0],
              [0,  0, 0,  0],
              [-1, 0, 1,  0],
              [0,  0, 0,  0]])
q2 = np.array([1, 0, 0, 0]).T
R2 = np.ones((1, 1))

x1 = np.array([1., 0, 1, 0])
horizon = 10

xs, us, nash_costs = LQFeedbackUnroll(A, B1, B2, Q1, Q2, q1, q2, R1, R2, x1, horizon)
