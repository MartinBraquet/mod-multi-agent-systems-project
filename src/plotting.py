from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def plotAnim2D(xs, sigmas, horizon, nx, muU, muV, sigmaU, sigmaV, fig_name, fig_subtitle, SIGMA):

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    
    color = 'b'
    ls = '--'
    lw = 1
    lnsx, = plt.plot([], [], color=color, marker='+')
    lnsigma, = plt.plot([], [], color=color, linestyle=ls, linewidth=lw, label='current')
    
    phi = np.linspace(0, 2*np.pi, 100)
    R = np.array([np.cos(phi),np.sin(phi)]).T
    
    state_mean_list = []
    for i in range(horizon+1):
        state_mean_list.append(xs[i*nx: (i+1)*nx])
    
    def init():
        ax.set_xlim(-1, 1.5)
        ax.set_ylim(-1, 1.5)
        return lnsx, lnsigma,
    
    def update(frame):
    
        Ytilde = SIGMA * R @ sqrtm(sigmas[frame])
        xdata = xs[0, frame] + Ytilde[:,0]
        ydata = xs[1, frame] + Ytilde[:,1]
        lnsigma.set_data(xdata, ydata)
        
        xdata = xs[0, frame]
        ydata = xs[1, frame]
        lnsx.set_data(xdata, ydata)
        
        return lnsx, lnsigma,
    
    
    ax.plot(xs[0,0], xs[1,0], color='c', marker='+')
    Ytilde = SIGMA * R @ sqrtm(sigmas[0])
    ax.plot(xs[0,0]+Ytilde[:,0], xs[1,0]+Ytilde[:,1], color='c', linewidth=lw, label='Initial')
    
    ax.plot(muU[0], muU[1], 'r+')
    Ytilde = SIGMA * R @ sqrtm(sigmaU)
    ax.plot(muU[0]+Ytilde[:,0], muU[1]+Ytilde[:,1], 'r-', label='Player U')

    ax.plot(muV[0], muV[1], 'g+')
    Ytilde = SIGMA * R @ sqrtm(sigmaV)
    ax.plot(muV[0]+Ytilde[:,0], muV[1]+Ytilde[:,1], 'g-', label='Player V')
    ax.legend(loc=2)
    
    anim = FuncAnimation(fig, update, frames=np.arange(horizon), init_func=init, blit=True)
    fig.suptitle(fig_subtitle, fontsize=14)
    
    anim.save(fig_name, writer='imagemagick', fps=60)
    
    plt.show()
    


def plot_2dsys(xs, sigmas, N, nx, nu, nv, muU, muV, sigmaU, sigmaV, SIGMA=2):

    phi = np.linspace(0, 2*np.pi, 100)
    R = np.array([np.cos(phi),np.sin(phi)]).T

    fig, ax = plt.subplots()

    ax.plot(xs[0,-1], xs[1,-1], 'go', markersize=4)

    for index, cov_matrix in enumerate(sigmas):
        if index == len(sigmas)-1:
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
        else:
            color = 'b'
            ls = '--'
            lw = 1
            label = None
            alpha = 0.1

        Ytilde = SIGMA * R @ sqrtm(cov_matrix)
        ax.plot(xs[0,index] + Ytilde[:,0],
                xs[1,index] + Ytilde[:,1], color=color,
                linestyle=ls,
                linewidth=lw, label=label)
        ax.plot(xs[0,index], xs[1,index], 'bo', markersize=4)

    ax.plot(muU[0], muU[1], 'r+')
    Ytilde = SIGMA * R @ sqrtm(sigmaU)
    ax.plot(muU[0]+Ytilde[:,0], muU[1]+Ytilde[:,1], 'r-', label='Player U')

    ax.plot(muV[0], muV[1], 'c+')
    Ytilde = SIGMA * R @ sqrtm(sigmaV)
    ax.plot(muV[0]+Ytilde[:,0], muV[1]+Ytilde[:,1], 'c-', label='Player V')
    ax.legend(loc=2)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')

    return fig, ax
