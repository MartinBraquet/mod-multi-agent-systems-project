import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

from covsteer_utils import solveCCPforU, solveCCPforV, unroll_dynamics_costs

from systems.double_integrator import integrator_cost, integrator_dynamics

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

prob_dynamics = integrator_dynamics
prob_cost = integrator_cost

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
CCP_iter = 20
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

    U_current, V_current = U_new, V_new

policy_change_u = np.array(policy_change_u)
policy_change_v = np.array(policy_change_v)

np.save(saveDataFolder + 'delta_u.npy', policy_change_u)
np.save(saveDataFolder + 'delta_v.npy', policy_change_v)

np.save(saveDataFolder + 'cost_u.npy', np.array(cost_u_list))
np.save(saveDataFolder + 'cost_v.npy', np.array(cost_v_list))

# set_trace()

Ex, Exx, cost_u, cost_v, mu_f, Sigma_f  = unroll_dynamics_costs(
                                            integrator_dynamics,
                                            integrator_cost,
                                            U_current,
                                            V_current)

traj_data = (Ex, Exx, N, nx, nu, nv,
             muU, muV, sigmaU, sigmaV,
             U_current, V_current)
np.save(saveDataFolder + 'traj_data.npy', traj_data)

# set_trace()
# policy_data = (U_current, V_current)
# np.save('policy_data.npy', policy_data)

# fig_u, ax_u = plt.subplots()
# ax_u.plot(policy_change_u[:,0], label=r'$\Delta \bar{U}$')
# ax_u.plot(policy_change_u[:,1], label=r'$\Delta L_{u}$')
# ax_u.plot(policy_change_u[:,2], label=r'$\Delta K_{u}$')
# ax_u.legend(prop={'size':12})
#
# fig_v, ax_v = plt.subplots()
# ax_v.plot(policy_change_v[:,0], label=r'$\Delta \bar{V}$')
# ax_v.plot(policy_change_v[:,1], label=r'$\Delta L_{v}$')
# ax_v.plot(policy_change_v[:,2], label=r'$\Delta K_{v}$')
# ax_v.legend(prop={'size':12})
#
# fig_u_half, ax_u_half = plt.subplots()
# ax_u_half.plot(policy_change_u[int(i/2):,0], label=r'$\Delta \bar{U}$')
# ax_u_half.plot(policy_change_u[int(i/2):,1], label=r'$\Delta L_{u}$')
# ax_u_half.plot(policy_change_u[int(i/2):,2], label=r'$\Delta K_{u}$')
# ax_u_half.legend(prop={'size':12})
#
# fig_v_half, ax_v_half = plt.subplots()
# ax_v_half.plot(policy_change_v[int(i/2):,0], label=r'$\Delta \bar{V}$')
# ax_v_half.plot(policy_change_v[int(i/2):,1], label=r'$\Delta L_{v}$')
# ax_v_half.plot(policy_change_v[int(i/2):,2], label=r'$\Delta K_{v}$')
# ax_v_half.legend(prop={'size':12})
#
# plt.show()
