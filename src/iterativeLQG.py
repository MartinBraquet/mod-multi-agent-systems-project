import numpy as np
from utils import regularizeW, unroll_OpenLoop, Wasserstein_Gaussian
from LQsolver.LQFeedbackSolverUnroll import LQFeedbackUnroll

def IterativeLQG(project):
    """
    This function computes the solution to the Iterative LQ game

    Inputs:
        project : Parameters of the system (dynamics, cost, initial condition and horizon)
    Outputs:
        xs       : State solution -> numpy.ndarray(nx, 1)
        us       : Control solution for player-1 -> numpy.ndarray(nx, 1)
        vs       : Control solution for player-2 -> numpy.ndarray(nx, 1)
        J1       : Solution cost for player-1 -> numpy.ndarray(1, 1)
        J2       : Solution cost for player-2 -> numpy.ndarray(1, 1)
    """
    dyn = project.dynamics
    cost = project.cost

    horizon = len(dyn.Alist)
    xDym = len(dyn.Alist[0])
    uDym = dyn.Blist[0].shape[1]
    N = 2

    # Initial control guesses (TO TWEAK)
    us = np.zeros((uDym,horizon))
    vs = np.zeros((uDym,horizon))
    Ps = [np.zeros((uDym, xDym, horizon)) for j in range(N)]
    a = [np.zeros((uDym, horizon)) for j in range(N)] 
    errorPs = [np.array([])] * N
    errorAs = [np.array([])] * N
    Q1 = [np.zeros((xDym,xDym))] * horizon
    Q2 = [np.zeros((xDym,xDym))] * horizon
    
    q1 = [np.zeros(xDym)] * horizon
    q2 = [np.zeros(xDym)] * horizon

    eps = 1e-2

    max_iter = 100
    for nIter in range(1,max_iter):
        # Derive rho_N* based on u1* and u2*
        mus, sigmas = unroll_OpenLoop(dyn, us, vs, Ps)
        muN, sigmaN = mus[:,-1], sigmas[-1]

        # Regularize W (LQ form w.r.t. u1 and u2)
        Q1[-1], q1[-1] = regularizeW(muN, sigmaN, cost.muU, cost.sigmaU)
        Q2[-1], q2[-1] = regularizeW(muN, sigmaN, cost.muV, cost.sigmaV)
        
        #print(cost.muU.T @ Q1[-1] @ cost.muU, cost.muV.T @ Q2[-1] @ cost.muV)

        # Find u1 and u2 by solving the LQG game via dynamic programming
        xs, sigmas, controls, Psnew, anew, nash_costs = LQFeedbackUnroll(dyn.Alist[0], dyn.Blist[0], dyn.Dlist[0], Q1, Q2, q1, q2, cost.Rulist[0], cost.Rvlist[0], dyn.mu0, dyn.sigma0, dyn.sigmaWlist[0], horizon)
        us, vs = controls

        # Until u1 and u2 converge
        convergedFlag = True
        for i in range(N):
            currErrorPs = np.linalg.norm(Ps[i] - Psnew[i])
            errorPs[i] = np.hstack([errorPs[i], currErrorPs])
            currErrorAs = np.linalg.norm(a[i] - anew[i])
            errorAs[i] = np.hstack([errorAs[i], currErrorAs])
            if currErrorPs > eps or currErrorAs > eps:
                convergedFlag = False
                
        if convergedFlag:
            print("Iterative LQG converged after", nIter, "iterations")
            break

        Ps, a = Psnew, anew
        
    if not convergedFlag:
        print("Iterative LQG dit not converge after", max_iter, "iterations")

    J1 = nash_costs[0]
    J2 = nash_costs[1]

    return xs, sigmas, us, vs, J1, J2, errorPs, errorAs
