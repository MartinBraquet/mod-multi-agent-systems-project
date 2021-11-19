import numpy as np
from utils import regularizeW, unroll_OpenLoop
from LQsolver/LQSolverFunction import LQFeedbackFunction

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
    mu0 = project.mu0
    sigma0 = project.sigma0
    dyn = project.dynamics
    cost = project.cost
    
    horizon = len(dyn.A)
    
    # Initial control guesses (TO TWEAK)
    us = [0] * horizon
    vs = [0] * horizon
    
    max_iter = 100
    convergedFlag = False
    for _ in range(max_iter):
        # Derive rho_N* based on u1* and u2*
        mus, sigmas = unroll_OpenLoop(dyn, mu0, sigma0, us, vs)
        muN, sigmaN = mus[-1], sigmas[-1]
        
        # Regularize W (LQ form w.r.t. u1 and u2)
        Q1, q1 = regularizeW(muN, sigmaN, cost.muU, cost.sigmaU)
        Q2, q2 = regularizeW(muN, sigmaN, cost.muV, cost.sigmaV)
        
        # Find u1 and u2 by solving the LQG game via dynamic programming
        xs, control, nash_costs = LQFeedbackFunction(dyn.A, dyn.B, dyn.D, Q1, Q2, q1, q2, cost.Rulist, cost.Rvlist, x0, horizon)
        usnew, vsnew = control
        
        # Until u1 and u2 converge
        if np.norm(us - usnew) < eps && np.norm(vs - vsnew) < eps:
            convergedFlag = True
            break
        
        us, vs = usnew, vsnew

    if not convergedFlag:
        print("Iterative LQG dit not converge after", max_iter, "iterations")

    J1 = nash_costs[0]
    J2 = nash_costs[1]
    
    return xs, us, vs, J1, J2
