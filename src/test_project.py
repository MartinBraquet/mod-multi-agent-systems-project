import numpy as np
import sys
sys.path.append('LQsolver')

from ProjectClass import Project, Dynamics, Cost
from iterativeLQG import IterativeLQG
import matplotlib.pyplot as plt

def state4D():
    horizon = 50
    dt = .05

    # Cost
    Ru = np.ones((1, 1))
    Rulist = [Ru] * horizon

    Rv = np.ones((1, 1))
    Rvlist = [Rv] * horizon

    # Desired final distributions
    mu0 = np.array([0, 0, 0, 0]) # -> Starts at [1,0]
    sigma0 = np.diag((.001, .001, .01, .01))

    muU = np.array([.5, 0, 0, 0]) # -> P1 steers to [0,0]
    sigmaU = np.diag((.3, .5, .06, .06))

    muV = np.array([0, .5, 0, 0]) # -> P2 steers to [1,1]
    sigmaV = np.diag((.3, .5, .06, .06))

    lambda_ = 1

    cost = Cost(Rulist, Rvlist, muU, muV, sigmaU, sigmaV, lambda_)

    # Dynamics
    A = np.array([[1,  0,  dt,   0],
                  [0,  1,   0,  dt],
                  [0,  0,   1,   0],
                  [0,  0,   0,   1]])
    B = np.array([0, 0, dt, dt], ndmin = 2).T
    D = np.array([0, 0, dt, dt], ndmin = 2).T

    Alist = [A] * horizon
    Blist = [B] * horizon
    Dlist = [D] * horizon

    sigmaW = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    sigmaWlist = [sigmaW] * horizon

    d = np.array([1], ndmin = 2).T
    dlist = [d] * horizon

    dynamics = Dynamics(Alist, Blist, Dlist, dlist, sigmaWlist, mu0, sigma0)

    project = Project(dynamics, cost)

    return project

def state1D():
    horizon = 50
    dt = .05

    # Cost
    Ru = np.array([[1]])
    Rulist = [Ru] * horizon

    Rv = np.array([[1]])
    Rvlist = [Rv] * horizon

    # Desired final distributions
    mu0 = np.array([[0]]) # -> Starts at [0]
    sigma0 = np.diag([.001])

    muU = np.array([[-1]]) # -> P1 steers to [1]
    sigmaU = np.diag([.05])

    muV = np.array([[1.5]]) # -> P2 steers to [0]
    sigmaV = np.diag([.05])

    lambda_ = 1

    cost = Cost(Rulist, Rvlist, muU, muV, sigmaU, sigmaV, lambda_)

    # Dynamics
    A = np.array([[1]])
    B = np.array([[.1]])
    D = np.array([[.1]])

    Alist = [A] * horizon
    Blist = [B] * horizon
    Dlist = [D] * horizon

    sigmaW = np.array([[1e-3]])
    sigmaWlist = [sigmaW] * horizon

    d = np.array([[1]])
    dlist = [d] * horizon

    dynamics = Dynamics(Alist, Blist, Dlist, dlist, sigmaWlist, mu0, sigma0)

    project = Project(dynamics, cost)

    return project


def main():


    # Build project
    #project = state4D()
    project = state4D()

    # Testing the Iterative LQR
    xs, sigmas, us, vs, J1, J2 = IterativeLQG(project)

    print(xs[:,-1])
    print(sigmas[-1])
    
    plt.plot(xs[0,:],xs[1,:])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    


if __name__ == "__main__":
    main()
