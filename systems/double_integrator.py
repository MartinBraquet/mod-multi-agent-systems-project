import numpy as np
from src.ProjectClass import Cost, Dynamics

dt = 0.1
N = 10
Ak = np.array([[0,1],[0,0]])*dt + np.eye(2)
Bk = np.array([[0], [1]])*dt
Dk = Bk
Wk = np.eye(2)*np.sqrt(dt)
zk = np.zeros((2,1))

nx, nu, nv = 2, 1, 1

Alist = [Ak for i in range(N)]
Blist = [Bk for i in range(N)]
Dlist = [Dk for i in range(N)]
zlist = [zk for i in range(N)]
sigmaWlist = [Wk for i in range(N)]

mu0, sigma0 = np.zeros((nx,1)), np.eye(nx)

integrator_dynamics = Dynamics(Alist, Blist, Dlist, zlist,
                               sigmaWlist, mu0, sigma0)

muU, muV = np.zeros((nu,1)), np.zeros((nv,1))

sigmaU, sigmaV = np.array([[2, 1.5],[1.5, 2]]), np.array([[2, -1.5],[-1.5, 2]])

Ru, Rv = np.eye(nu), np.eye(nv)

lambda_ = 1000.

Rulist, Rvlist = [Ru for i in range(N)], [Rv for i in range(N)]

integrator_cost = Cost(Rulist, Rvlist, muU, muV, sigmaU, sigmaV, lambda_)
