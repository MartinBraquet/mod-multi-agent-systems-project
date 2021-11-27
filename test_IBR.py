import numpy as np
import cvxpy as cp

from src.covsteer_utils import solveCCPforU, solveCCPforV

from sys.double_integrator import integrator_cost, integrator_dynamics

def is_policy_close(policy_new, policy_prev, eps=1e-3):
    """
    Check whether the policies between consecutive iterations are close to each
    other with respect to the given tolerance value eps > 0.
    """
    ff_check = (np.abs(policy_new[0]-policy_prev[0]) <= eps).all()
    L_check = (np.abs(policy_new[1]-policy_prev[1]) <= eps).all()
    K_check = (np.abs(policy_new[2]-policy_prev[2]) <= eps).all()
    return ff_check and L_check and K_check

# Set problem dynamics and cost, It can be changed by importing a dynamics and
# cost structures from sys folder.
prob_dynamics = integrator_dynamics
prob_cost = integrator_cost

# Problem horizon:
N = len(prob_dynamics.Alist)
# State, control_u and control_v dimensions:
nx = prob_dynamics.Alist[0].shape[1]
nu = prob_dynamics.Blist[0].shape[1]
nv = prob_dynamics.Dlist[0].shape[1]

# Initialize both player's policy to zeros
Ubar_init, Lu_init, Ku_init = (np.zeros((nu*N, 1)),
                               np.zeros((nu*N, nx)),
                               np.zeros((nu*N, nx*(N+1))))
Vbar_init, Lv_init, Kv_init = (np.zeros((nu*N, 1)),
                               np.zeros((nu*N, nx)),
                               np.zeros((nu*N, nx*(N+1))))

IBR_convergence_flag = False
eps_conv = 1e-4
N_IBR = 10
for i in range(N_IBR):

    Upolicy_new, conv_CCP, costU, costV = solveCCPforU(
        prob_dynamics, prob_cost,
        Upolicy, Vpolicy,
        CCP_iter=10, solver="MOSEK", eps=1e-4)

    Vpolicy_new, conv_CCP, costV, costU = solveCCPforV(
        prob_dynamics, prob_cost,
        Upolicy, Vpolicy,
        CCP_iter=10, solver="MOSEK", eps=1e-4)

    if (is_policy_close(Upolicy_new, Upolicy_prev, eps_conv) and
        is_policy_close(Vpolicy_new, Vpolicy_prev, eps_conv) ):
        IBR_convergence_flag = True

    pass
