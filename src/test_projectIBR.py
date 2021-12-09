import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

from covsteer_utils import solveCCPforU, solveCCPforV, unroll_dynamics_costs

from systems.double_integrator import integrator_cost, integrator_dynamics
from systems.random_system import rand_cost, rand_dynamics

from pdb import set_trace

saveDataFolder = 'IBR data/'

def is_policy_close(policy_new, policy_prev, eps=1e-2):
    """
    Check whether the policies between consecutive iterations are close to each
    other with respect to the given tolerance value eps > 0.
    """
    ff_check = (np.abs(policy_new[0]-policy_prev[0]) <= eps).all()
    L_check = (np.abs(policy_new[1]-policy_prev[1]) <= eps).all()
    K_check = (np.abs(policy_new[2]-policy_prev[2]) <= eps).all()
    return ff_check and L_check and K_check

plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["text.usetex"] = True

# prob_dynamics = integrator_dynamics
# prob_cost = integrator_cost
prob_dynamics = rand_dynamics
prob_cost = rand_cost

# Problem horizon:
N = len(prob_dynamics.Alist)
# State, control_u and control_v dimensions:
nx = prob_dynamics.Alist[0].shape[1]
nu = prob_dynamics.Blist[0].shape[1]
nv = prob_dynamics.Dlist[0].shape[1]

# Desired Distributions:
muU, muV = prob_cost.muU, prob_cost.muV
sigmaU, sigmaV = prob_cost.sigmaU, prob_cost.sigmaV

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

N_ibr = 1000
U_current, V_current = U_init, V_init
CCP_iter = 10
cost_u_list, cost_v_list = [], []
policy_change_u, policy_change_v = [], []

for i in range(N_ibr):
    print('Iteration {}:'.format(i))
    print('-Update U-')
    U_new, conv_u, cost_u, cost_v  = solveCCPforU(
                                        prob_dynamics, prob_cost,
                                        U_current,
                                        V_current,
                                        CCP_iter=CCP_iter)
    cost_u_list.append(cost_u)
    cost_v_list.append(cost_v)

    print('Cost U : {}'.format(cost_u))
    print('Cost V : {}'.format(cost_v))
    print('-Update V-')

    V_new, conv_v, cost_v, cost_u  = solveCCPforV(
                                        prob_dynamics, prob_cost,
                                        U_new,
                                        V_current,
                                        CCP_iter=CCP_iter)
    cost_u_list.append(cost_u)
    cost_v_list.append(cost_v)

    print('Cost U : {}'.format(cost_u))
    print('Cost V : {}'.format(cost_v))
    print('U policy close : {}'.format(is_policy_close(U_new, U_current) ) )
    print('V policy close : {}'.format(is_policy_close(V_new, V_current) ) )
    print('----')

    Ubar_diff = np.linalg.norm(U_new[0]-U_current[0], 2)
    Vbar_diff = np.linalg.norm(V_new[0]-V_current[0], 2)
    Lu_diff = np.linalg.norm(U_new[1]-U_current[1], 2)
    Lv_diff = np.linalg.norm(V_new[1]-V_current[1], 2)
    Ku_diff = np.linalg.norm(U_new[2]-U_current[2], 2)
    Kv_diff = np.linalg.norm(V_new[2]-V_current[2], 2)

    policy_change_u.append([Ubar_diff, Lu_diff, Ku_diff])
    policy_change_v.append([Vbar_diff, Lv_diff, Kv_diff])

    if is_policy_close(U_new, U_current) and is_policy_close(V_new, V_current):
        print('Converged in {} steps'.format(i+1))
        break
    # if i == 79:
    #     pass # Just for 1 example to show that IBR diverges
    #     break

    U_current, V_current = U_new, V_new

policy_change_u = np.array(policy_change_u)
policy_change_v = np.array(policy_change_v)

np.save(saveDataFolder + 'delta_u.npy', policy_change_u)
np.save(saveDataFolder + 'delta_v.npy', policy_change_v)

np.save(saveDataFolder + 'cost_u.npy', np.array(cost_u_list))
np.save(saveDataFolder + 'cost_v.npy', np.array(cost_v_list))

Ex, Exx, cost_u, cost_v, mu_f, Sigma_f  = unroll_dynamics_costs(
                                            prob_dynamics,
                                            prob_cost,
                                            U_current,
                                            V_current)

traj_data = (Ex, Exx, N, nx, nu, nv,
             muU, muV, sigmaU, sigmaV,
             U_current, V_current)
np.save(saveDataFolder + 'traj_data.npy', traj_data)
