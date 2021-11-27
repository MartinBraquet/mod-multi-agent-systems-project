import numpy as np
from Utils import udim, xdim, Dynamics, Cost, add_control_cost, unroll_feedback, evaluate

# Solve a finite horizon, discrete time LQ game to feedback Nash equilibrium.
# Returns feedback matrices P[player][:, :, time]
def solve_lq_feedback(dyn, costs, horizon):
    N = len(dyn.Bs)

    P = [np.zeros((udim(dyn,j), xdim(dyn), horizon)) for j in range(N)]
    a = [np.zeros((udim(dyn,j), horizon)) for j in range(N)]

    Z = [np.zeros((xdim(dyn), xdim(dyn))) for j in range(N)]
    zeta = [np.zeros(xdim(dyn)) for j in range(N)]

    for t in np.arange(horizon-1,-1,-1):

		# M * P = Bp and M * a = Ba
        M = np.array([])
        for i in range(N):
            Mcolcat = np.array([])
            for j in range(N):
                Mij = dyn.Bs[i].T @ Z[i] @ dyn.Bs[j]
                if j == i:
                    Mij = Mij + costs[i].Rs[i]
                Mcolcat = np.hstack([Mcolcat, Mij]) if Mcolcat.size else Mij
            M = np.vstack([M,Mcolcat]) if M.size else Mcolcat

        Bp = np.array([])
        Ba = np.array([])
        for i in range(N):
            Bpi = dyn.Bs[i].T @ Z[i] @ dyn.A
            Bp = np.vstack((Bp,Bpi)) if Bp.size else Bpi

            Bai = dyn.Bs[i].T @ zeta[i]
            Ba = np.vstack((Ba,Bai)) if Ba.size else Bai

        Psol = np.linalg.solve(M, Bp)
        asol = np.linalg.solve(M, Ba)

        nusum = 0
        for i in range(N):
            nu = udim(dyn,i)
            P[i][:,:,t] = Psol[nusum:nusum+nu,:]
            a[i][:,t] = asol[nusum:nusum+nu]
            nusum += nu

        F = dyn.A - sum(dyn.Bs[j] @ P[j][:,:,t] for j in range(N))
        beta = - sum(dyn.Bs[j] @ a[j][:,t] for j in range(N))

        Z = [costs[i].Q[t] + sum([P[j][:,:,t].T @ costs[i].Rs[j] @ P[j][:,:,t] for j in costs[i].Rs]) + F.T @ Z[i] @ F for i in range(N)]
        zeta = [costs[i].q[t] + sum([P[j][:,:,t].T @ costs[i].Rs[j] @ a[j][:,t] for j in costs[i].Rs]) + F.T @ (zeta[i] + Z[i] @ beta) for i in range(N)]

    return P, a

def LQFeedbackUnroll(A, B1, B2, Q1, Q2, q1, q2, R1, R2, mu0, sigma0, sigmaW, horizon):

    dyn = Dynamics(A, [B1, B2], mu0, sigma0, sigmaW)

    c1 = Cost(Q1, q1)
    add_control_cost(c1, 0, R1)

    c2 = Cost(Q2, q2)
    add_control_cost(c2, 1, R2)

    costs = [c1, c2]

    # Ensure that the feedback solution satisfies Nash conditions of optimality
    # for each player, holding others' strategies fixed.
    Ps, a = solve_lq_feedback(dyn, costs, horizon)
    xs, sigmas, us = unroll_feedback(dyn, Ps, a)
    nash_costs = [evaluate(c, xs, us) for c in costs]

    return xs, sigmas, us, nash_costs
