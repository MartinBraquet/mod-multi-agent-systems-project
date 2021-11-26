import numpy as np

from ProjectClass import Project, Dynamics, Cost
from IterativeLQG import IterativeLQG


def main():
    horizon = 50
    dt = .05

    # Cost
    Ru = np.ones((1, 1))
    Rulist = [Ru] * horizon

    Rv = np.ones((1, 1))
    Rvlist = [Rv] * horizon

    # Desired final distributions
    mu0 = np.array([1, 0, 0, 0]) # -> Starts at [1,0]
    sigma0 = np.diag((.001, .001, .01, .01))

    muU = np.array([1, 0, 0, 0]) # -> P1 steers to [0,0]
    sigmaU = np.diag((.5, .5, .1, .1))

    muV = np.array([1, 0, 0, 0]) # -> P2 steers to [1,1]
    sigmaV = np.diag((.5, .5, .1, .1))

    lambda_ = 1

    cost = Cost(Rulist, Rvlist, muU, muV, sigmaU, sigmaV, lambda_)

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

    sigmaW = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    sigmaWlist = [sigmaW] * horizon

    d = np.array([1], ndmin = 2).T
    dlist = [d] * horizon

    dynamics = Dynamics(Alist, Blist, Dlist, dlist, sigmaWlist, mu0, sigma0)

    # Build project
    project = Project(dynamics, cost)

    # Testing the Iterative LQR
    xs, us, vs, J1, J2 = IterativeLQG(project)


if __name__ == "__main__":
    main()
