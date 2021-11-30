import numpy as np
from matplotlib import pyplot as plt
from src.plotting import plot_2dsys

from pdb import set_trace

plt.rcParams["text.usetex"] = True
plt.rcParams["savefig.bbox"] = 'tight'
# plt.rcParams[]

policy_change_u = np.load('delta_u.npy')
policy_change_v = np.load('delta_v.npy')

cost_u, cost_v = np.load('cost_u.npy'), np.load('cost_v.npy')

fig_costu, ax_costu = plt.subplots()
ax_costu.plot(cost_u)
ax_costu.set_title(r'$J_{u}^{\star}$')

fig_costv, ax_costv = plt.subplots()
ax_costv.plot(cost_v)
ax_costv.set_title(r'$J_{v}^{\star}$')

fig_costu.savefig('Cost_u_fig.png')
fig_costv.savefig('Cost_v_fig.png')
# set_trace()

fig_u, ax_u = plt.subplots()
ax_u.plot(policy_change_u[:,0], label=r'$\Delta \bar{U}$')
ax_u.plot(policy_change_u[:,1], label=r'$\Delta L_{u}$')
ax_u.plot(policy_change_u[:,2], label=r'$\Delta K_{u}$')
ax_u.set_xlabel('Iteration')
ax_u.set_ylabel('Policy Change')
ax_u.legend(prop={'size':12})

fig_v, ax_v = plt.subplots()
ax_v.plot(policy_change_v[:,0], label=r'$\Delta \bar{V}$')
ax_v.plot(policy_change_v[:,1], label=r'$\Delta L_{v}$')
ax_v.plot(policy_change_v[:,2], label=r'$\Delta K_{v}$')
ax_u.set_xlabel('Iteration')
ax_u.set_ylabel('Policy Change')
ax_v.legend(prop={'size':12})

fig_u.savefig('Uconv.png')
fig_v.savefig('Vconv.png')

len_ = policy_change_u.shape[0]
len_half = int(len_/2)
dummy_list = [i for i in range(len_half,len_)]

fig_u_half, ax_u_half = plt.subplots()
ax_u_half.plot(dummy_list, policy_change_u[len_half:,0], label=r'$\Delta \bar{U}$')
ax_u_half.plot(dummy_list, policy_change_u[len_half:,1], label=r'$\Delta L_{u}$')
ax_u_half.plot(dummy_list, policy_change_u[len_half:,2], label=r'$\Delta K_{u}$')
ax_u.set_xlabel('Iteration')
ax_u.set_ylabel('Policy Change')
ax_u_half.legend(prop={'size':12})

fig_v_half, ax_v_half = plt.subplots()
ax_v_half.plot(dummy_list, policy_change_v[len_half:,0], label=r'$\Delta \bar{V}$')
ax_v_half.plot(dummy_list, policy_change_v[len_half:,1], label=r'$\Delta L_{v}$')
ax_v_half.plot(dummy_list, policy_change_v[len_half:,2], label=r'$\Delta K_{v}$')
ax_u.set_xlabel('Iteration')
ax_u.set_ylabel('Policy Change')
ax_v_half.legend(prop={'size':12})

fig_u_half.savefig('Uconv_half.png')
fig_v_half.savefig('Vconv_half.png')

traj_data = np.load('traj_data.npy', allow_pickle=True)

# print(len(traj_data))

(Ex, Exx, N, nx, nu, nv,
    muU, muV, sigmaU, sigmaV,
    U_current, V_current) = traj_data

fig_traj, ax_traj = plot_2dsys(Ex, Exx, N, nx, nu, nv,
                               muU, muV, sigmaU, sigmaV)
fig_traj.savefig('traj.png')


plt.show()
