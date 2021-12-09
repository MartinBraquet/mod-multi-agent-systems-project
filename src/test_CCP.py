import numpy as np
import cvxpy as cp

from covsteer_utils import solveCCPforU, solveCCPforV
from covsteer_utils import getMatirces
from covsteer_utils import setDecisionVariables

from systems.double_integrator import integrator_cost, integrator_dynamics

from pdb import set_trace

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
                               np.zeros((nu*N, nx*N)))
Vbar_init, Lv_init, Kv_init = (np.zeros((nu*N, 1)),
                               np.zeros((nu*N, nx)),
                               np.zeros((nu*N, nx*N)))

U_init = (Ubar_init, Lu_init, Ku_init)
V_init = (Vbar_init, Lv_init, Kv_init)

# Gamma, Hu, Hv, Hw, Z, Wbig, Rubig, Rvbig = getMatirces(prob_dynamics, prob_cost)

# vars = setDecisionVariables(prob_dynamics)

# set_trace()


Upolicy_value, convergence_flag, total_cost_u, total_cost_v  = solveCCPforU(
                                    prob_dynamics, prob_cost,
                                    U_init,
                                    V_init,
                                    CCP_iter=20, solver="MOSEK", eps=1e-4)

Vpolicy_value, convergence_flag, total_cost_v, total_cost_u  = solveCCPforV(
                                    prob_dynamics, prob_cost,
                                    U_init,
                                    V_init,
                                    CCP_iter=20, solver="MOSEK", eps=1e-4)
