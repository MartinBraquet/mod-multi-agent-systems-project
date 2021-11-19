import numpy as np

from ProjectClass import Project, Dynamics, Cost
from IterativeLQG import IterativeLQG


def main():
    horizon = 50
    dt = .1
    
    # Cost
    Ru = np.ones((1, 1))
    Rulist = [Ru] * horizon

    Rv = np.ones((1, 1))
    Rvlist = [Rv] * horizon
    
    # Desired final distributions
    mu0 = np.array([1, 0, 0, 0]) # -> Starts at [1,0]
    sigma0 = np.ones((1, 1))
    
    muU = np.array([0, 0, 0, 0]) # -> P1 steers to [0,0]
    sigmaU = np.ones((1, 1))
    
    muV = np.array([1, 1, 0, 0]) # -> P2 steers to [1,1]
    sigmaV = np.ones((1, 1))
    
    cost = Cost(Rulist, Rvlist, muU, muV, sigmaU, sigmaV)
    
    # Dynamics    
    A = np.array([[1,  0,  dt,   0],
                  [0,  1,   0,  dt],
                  [0,  0,   1,   0],
                  [0,  0,   0,   1]])
    B = np.array([0, 0, dt, 0], ndmin = 2).T
    D = np.array([0, 0, 0, dt], ndmin = 2).T
    
    Alist = [A] * horizon
    Blist = [B] * horizon
    Dlist = [D] * horizon
    
    sigmaW = np.ones((0.01, 0.01, 0.01, 0.01))
    sigmaWlist = [sigmaW] * horizon
    
    dynamics = Dynamics(Alist, Blist, Dlist, sigmaWlist)

    # Build project
    project = Project(mu0, sigma0, dynamics, cost)

    # Testing the Iterative LQR
    xs, us, vs, J1, J2 = IterativeLQG(project)


if __name__ == "__main__":
    main()
