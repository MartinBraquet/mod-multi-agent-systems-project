import numpy as np
from Utils import * 
# Solve a finite horizon, discrete time LQ game to feedback Nash equilibrium.
# Returns feedback matrices P[player][:, :, time]
def solve_lq_feedback(dyn, costs, horizon):
    N = len(dyn.Bs)

    P = [np.zeros((udim(dyn,j), xdim(dyn), horizon)) for j in range(N)]

    Z = [np.zeros((xdim(dyn), xdim(dyn))) for j in range(N)]

    for t in np.arange(horizon-1,-1,-1):

        M = None
        for i in range(N):
            Mcolcat = None
            for j in range(N):
                Mij = dyn.Bs[i].T @ Z[i] @ dyn.Bs[j]
                if j == i:
                    Mij = Mij + costs[i].Rs[i]

                if Mcolcat is None:
                    Mcolcat = Mij
                else:
                    Mcolcat = np.concatenate((Mcolcat, Mij), axis=1)
                    
            if M is None:
                M = Mcolcat
            else:
                M = np.concatenate((M,Mcolcat), axis=0)

        O = None
        for i in range(N):
            Op = dyn.Bs[i].T @ Z[i] @ dyn.A
            if O is None:
                O = Op
            else:
                O = np.concatenate((O,Op), axis=0)

        Psol = np.linalg.solve(M, O)

        nusum = 0
        for i in range(N):
            nu = udim(dyn,i)
            P[i][:,:,t] = Psol[nusum:nusum+nu,:]
            nusum += nu

        F = dyn.A - sum(dyn.Bs[j] @ P[j][:,:,t] for j in range(N))
        Z = [F.T @ Z[i] @ F + costs[i].Q + sum([P[j][:,:,t].T @ costs[i].Rs[j] @ P[j][:,:,t] for j in costs[i].Rs]) for i in range(N)]
    

    return P
