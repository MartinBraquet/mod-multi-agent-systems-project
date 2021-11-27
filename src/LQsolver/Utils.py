import numpy as np
# Utilities for Assignment 3. You should not need to modify this file.

# Cost for a single player.
# Form is: x^T_t Q^i x + \sum_j u^{jT}_t R^{ij} u^j_t.
# For simplicity, assuming that Q, R are time-invariant, and that dynamics are
# linear time-invariant, i.e. x_{t+1} = A x_t + \sum_i B^i u^i_t.
class Cost():
    def __init__(self, Q, q, Rs):
        self.Q = Q
        self.q = q
        self.Rs = Rs
    def __init__(self, Q, q):
        self.Q = Q
        self.q = q
        self.Rs = dict()

#Cost(Q) = Cost(Q, Dict{Int, Matrix{eltype(Q)}}())

# Method to add R^{ij}s to a Cost struct.
def add_control_cost(c, other_player_idx, Rij):
    c.Rs[other_player_idx] = Rij

# Evaluate cost on a state/control trajectory.
# - xs[:, time]
# - us[player][:, time]
def evaluate(c, xs, us):
    horizon = xs.shape[1]

    total = 0.0
    for tt in range(horizon):
        total += (xs[:, tt].T @ c.Q[tt] + 2 * c.q[tt].T) @ xs[:, tt]
        total += sum(us[jj][:, tt].T @ c.Rs[jj] @ us[jj][:, tt] for jj in c.Rs)

    return total

# Dynamics.
# linear time-invariant, i.e. x_{t+1} = A x_t + \sum_i B^i u^i_t.
class Dynamics():
    def __init__(self, A, Bs, mu0, sigma0, sigmaW):
        self.A = A
        self.Bs = Bs
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigmaW = sigmaW

def xdim(dyn):
    return np.size(dyn.A,1)

def udim(dyn):
    return sum(B.shape[1] for B in dyn.Bs)

def udim(dyn, player_idx):
    return dyn.Bs[player_idx].shape[1]

# Function to unroll a set of feedback matrices from an initial condition.
# Output is a sequence of states xs[:, time] and controls us[player][:, time].
def unroll_feedback(dyn, Ps, a):
    assert len(dyn.mu0) == xdim(dyn)

    N = len(Ps)
    assert N == len(dyn.Bs)

    horizon = Ps[0].shape[2]

    # Populate state/control trajectory.
    xs = np.zeros((xdim(dyn), horizon))
    xs[:, 0] = dyn.mu0
    sigmas = [np.zeros((N, N))] * horizon
    sigmas[0] = dyn.sigma0
    us = [np.zeros((udim(dyn, ii), horizon)) for ii in range(N)]
    for tt in np.arange(1,horizon):
        for ii in range(N):
            us[ii][:, tt - 1] = -Ps[ii][:, :,tt-1] @ xs[:,tt-1] - a[ii][:,tt-1]

        w = np.random.normal(np.zeros(xdim(dyn)), dyn.sigmaW)
        xs[:, tt] = dyn.A @ xs[:,tt-1] + sum(dyn.Bs[ii] @ us[ii][:,tt-1] for ii in range(N)) + w
        sigmas[tt] = dyn.A @ sigmas[tt-1] @ dyn.A.T + dyn.sigmaW

    # Controls at final time.
    for ii in range(N):
        us[ii][:, horizon-1] = -Ps[ii][:, :, horizon-1] @ xs[:, horizon-1] - a[ii][:,horizon-1]

    return xs, sigmas, us
