from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
import numpy as np


def plot_2dsys(Ex, Exx, N, nx, nu, nv, muU, muV, sigmaU, sigmaV, SIGMA=2):
    state_mean_list = []
    state_cov_list = []

    for i in range(N+1):
        state_mean_list.append(Ex[i*nx: (i+1)*nx])
        state_cov_list.append(Exx[i*nx:(i+1)*nx, i*nx:(i+1)*nx])
    # pdb.set_trace()

    phi = np.linspace(0, 2*np.pi, 100)
    R = np.array([np.cos(phi),np.sin(phi)]).T

    fig, ax = plt.subplots()

    Ex_array = Ex.reshape(N+1, nx).T
    ax.plot(Ex_array[0][:-1], Ex_array[1][:-1], 'bo', markersize=4)
    ax.plot(Ex_array[0][-1], Ex_array[1][-1], 'go', markersize=4)

    for index, cov_matrix in enumerate(state_cov_list):
        color = 'b'
        ls = '--'
        lw = 1
        label = None
        alpha = 0.1
        if index == len(state_cov_list)-1:
            color = 'g'
            ls = '-'
            lw = 3
            label = 'Final'
            alpha = 1.0
        elif index == 0:
            color = 'orange'
            ls = '-'
            lw = 2
            label = 'Initial'
            alpha = .5

        # cov_matrix_pos = cov_matrix[0:2,0:2]
        Ytilde = SIGMA * R @ sqrtm(cov_matrix)
        ax.plot(Ex_array[0, index] + Ytilde[:,0],
                Ex_array[1, index] + Ytilde[:,1], color=color,
                linestyle=ls,
                linewidth=lw, label=label)

    ax.plot(muU[0], muU[1], 'r+')
    Ytilde = SIGMA * R @ sqrtm(sigmaU)
    ax.plot(muU[0]+Ytilde[:,0], muU[1]+Ytilde[:,1], 'r-', label='Player U')

    ax.plot(muV[0], muV[1], 'c+')
    Ytilde = SIGMA * R @ sqrtm(sigmaV)
    ax.plot(muU[0]+Ytilde[:,0], muU[1]+Ytilde[:,1], 'c-', label='Player V')
    ax.legend(loc=2)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')

    return fig, ax
