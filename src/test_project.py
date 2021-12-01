import numpy as np
import sys
sys.path.append('LQsolver')

from ProjectClass import Project, Dynamics, Cost
from iterativeLQG import IterativeLQG
from plotting import plot_2dsys
import matplotlib.pyplot as plt

def state4D():
    horizon = 100
    dt = .05

    # Cost
    Ru = np.diag((1, 1))
    Rulist = [Ru] * horizon

    Rv = np.diag((1, 1))
    Rvlist = [Rv] * horizon

    # Desired final distributions
    mu0 = np.array([0, 0, 0, 0]) # -> Starts at [1,0]
    sigma0 = np.diag((.01, .01, .01, .01))

    muU = np.array([0, 0, 0, 0]) # -> P1 steers to [0,0]
    sigmaU = np.diag((.01, .01, .01, .01))

    muV = np.array([4, 4, 0, 0]) # -> P2 steers to [1,1]
    sigmaV = np.diag((.2, .9, .06, .06))

    lambda_ = 1

    cost = Cost(Rulist, Rvlist, muU, muV, sigmaU, sigmaV, lambda_)

    # Dynamics
    A = np.array([[1,  0,  dt,   0],
                  [0,  1,   0,  dt],
                  [0,  0,   1,   0],
                  [0,  0,   0,   1]])
    
    D1control = False
    if D1control:
        B = np.array([0, 0, dt, 0], ndmin=2).T
        D = np.array([0, 0, 0, dt], ndmin=2).T
    else:
        B = np.array([[0, 0, dt, 0], [0, 0, 0, dt]]).T
        D = np.array([[0, 0, dt, 0], [0, 0, 0, dt]]).T

    Alist = [A] * horizon
    Blist = [B] * horizon
    Dlist = [D] * horizon

    sigmaW = np.diag([1e-3, 1e-3, 1e-3, 1e-3])
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

    project4D = True

    # Build project
    if project4D:
        project = state4D()
    else:
        project = state1D()

    # Testing the Iterative LQR
    xs, sigmas, us, vs, J1, J2, errorPs, errorAs = IterativeLQG(project)

    print(xs[:,-1])
    print(sigmas[-1])
    
    if project4D:
        c = project.cost
        xsStacked = np.array([])
        sigmas2D = []
        for t in range(xs.shape[1]):
            xsStacked = np.hstack([xsStacked,xs[0:2,t]])
            sigmas2D.append(sigmas[t][0:2,0:2])
        horizon = len(sigmas)-1; nx = 2; nu = 2; nv = 2        
        plot_2dsys(xsStacked, sigmas2D, horizon, nx, nu, nv, c.muU[0:2], c.muV[0:2], c.sigmaU[0:2,0:2], c.sigmaV[0:2,0:2], 1)
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.show()
        
        plt.plot(errorPs[0])
        plt.plot(errorPs[1])
        plt.plot(errorAs[0])
        plt.plot(errorAs[1])
        plt.xlabel('# Iterations')
        plt.ylabel('Error')
        plt.legend((r'$P^*$ - P1', r'$P^*$ - P2', r'$\alpha^*$ - P1', r'$\alpha^*$ - P2'))
        plt.show()
    


if __name__ == "__main__":
    main()
