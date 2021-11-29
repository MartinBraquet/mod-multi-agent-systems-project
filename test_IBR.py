import numpy as np
import cvxpy as cp

from src.covsteer_utils import solveCCPforU, solveCCPforV

from systems.double_integrator import integrator_cost, integrator_dynamics

from pdb import set_trace

def is_policy_close(policy_new, policy_prev, eps=1e-2):
    """
    Check whether the policies between consecutive iterations are close to each
    other with respect to the given tolerance value eps > 0.
    """
    ff_check = (np.abs(policy_new[0]-policy_prev[0]) <= eps).all()
    L_check = (np.abs(policy_new[1]-policy_prev[1]) <= eps).all()
    K_check = (np.abs(policy_new[2]-policy_prev[2]) <= eps).all()
    return ff_check and L_check and K_check

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
#
# vars = setDecisionVariables(prob_dynamics)

N_ibr = 100
U_current, V_current = U_init, V_init
CCP_iter = 10
for i in range(N_ibr):
    U_new, conv_u, cost_u, cost_v  = solveCCPforU(
                                        prob_dynamics, prob_cost,
                                        U_current,
                                        V_current,
                                        CCP_iter=CCP_iter)

    V_new, conv_v, cost_v, cost_u  = solveCCPforV(
                                        prob_dynamics, prob_cost,
                                        U_current,
                                        V_current,
                                        CCP_iter=CCP_iter)

    print('Cost U : {}'.format(cost_u))
    print('Cost V : {}'.format(cost_v))
    print('U policy close : {}'.format(is_policy_close(U_new, U_current) ) )
    print('V policy close : {}'.format(is_policy_close(V_new, V_current) ) )

    if is_policy_close(U_new, U_current) and is_policy_close(V_new, V_current):
        print('Converged in {} steps'.format(i+1))
        break

    U_current, V_current = U_new, V_new
