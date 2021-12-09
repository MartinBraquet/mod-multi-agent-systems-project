import numpy as np
from ProjectClass import Cost, Dynamics

dt = 0.1
N = 10
Ak = np.array([[0.0,1.0],[2.,3.]])*dt + np.eye(2)
Bk = np.array([[0.0], [1.]])*dt
Dk = np.array([[0.0], [1.]])*dt
Wk = 1.0*np.eye(2)*np.sqrt(dt)
zk = np.zeros((2,1))

nx, nu, nv = 2, 1, 1

mu0, sigma0 = np.zeros((nx,1)), 1.*np.eye(nx)

muU, muV = 0.*np.array([[0.5], [0.0]]), 0.*np.array([[0.8], [0.0]])

sigmaU, sigmaV = (
                np.eye(2),
                1.2*np.eye(2)
)
Ru, Rv = np.eye(nu), np.eye(nv)

lambda_ = 5000.

Alist = [Ak for i in range(N)]
Blist = [Bk for i in range(N)]
Dlist = [Dk for i in range(N)]
zlist = [zk for i in range(N)]
sigmaWlist = [Wk for i in range(N)]

rand_dynamics = Dynamics(Alist, Blist, Dlist, zlist,
                               sigmaWlist, mu0, sigma0)


Rulist, Rvlist = [Ru for i in range(N)], [Rv for i in range(N)]

rand_cost = Cost(Rulist, Rvlist, muU, muV, sigmaU, sigmaV, lambda_)
