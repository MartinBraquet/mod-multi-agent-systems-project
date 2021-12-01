import numpy as np
from scipy.linalg import sqrtm

def regularizeW(mu_f, Sigma_f, mu_d, Sigma_d):
    """
    This function linearizes the Wasserstein distance cost term around the
    mean and covariance of the terminal state of the previous solution. The
    outputs can be used as a quadratic cost function for the LQG game problem.
    Terminal State Cost: x^T Q x + 2 q^T x (+ cst term).

    Inputs:
        mu_f    : Terminal Mean Vector -> numpy.ndarray(nx, 1)
        Sigma_f : Terminal Covariance Matrix -> numpy.ndarray(nx, nx)
        mu_d    : Desired Mean Vector -> numpy.ndarray(nx, 1)
        Sigma_d : Desired Covariance Matrix -> numpy.ndarray(nx, nx)
    Outputs:
        Q       : Quadratic Cost Matrix -> numpy.ndarray(nx, nx)
        q       : Linear Cost Vector -> numpy.ndarray(nx, 1)
    """
    nx = mu_f.shape[0]
    Inx = np.eye(nx)
    Sigma_dsqrt = sqrtm(Sigma_d)
    Mhelp = np.linalg.inv(sqrtm(Sigma_dsqrt @ Sigma_f @ Sigma_dsqrt))
    M = Sigma_dsqrt @ Mhelp @ Sigma_dsqrt
    Q = Inx - M
    q = (M @ mu_f) / 2 - mu_d
    return Q, q


def Wasserstein_Gaussian(mu_f, Sigma_f, mu_d, Sigma_d):
    """
    This function computes the squared Wasserstein distance cost term

    Inputs:
        mu_f    : Terminal Mean Vector -> numpy.ndarray(nx, 1)
        Sigma_f : Terminal Covariance Matrix -> numpy.ndarray(nx, nx)
        mu_d    : Desired Mean Vector -> numpy.ndarray(nx, 1)
        Sigma_d : Desired Covariance Matrix -> numpy.ndarray(nx, nx)
    Outputs:
        dist    : squared Wasserstein distance -> scalar
    """
    Sigma_dsqrt = sqrtm(Sigma_d)
    W2 = np.linalg.norm(mu_f - mu_d)**2 + np.trace(Sigma_f + Sigma_d
                       - 2* sqrtm(Sigma_dsqrt @ Sigma_f @ Sigma_dsqrt))

    return W2

# Function to unroll the open-loop control inputs
def unroll_OpenLoop(dyn, us, vs, Ps):

    A = dyn.Alist
    B = dyn.Blist
    D = dyn.Dlist
    sigmaW = dyn.sigmaWlist
    mu0 = dyn.mu0
    sigma0 = dyn.sigma0

    N = len(mu0)
    horizon = len(A)

    # Populate state/control trajectory.
    mus = np.zeros((N, horizon))
    mus[:, 0] = mu0
    sigmas = [np.zeros((N, N))] * horizon
    sigmas[0] = sigma0
    for t in np.arange(1,horizon):
        #w = np.random.normal(np.zeros(N), sigmaW[t])
        mus[:, t] = A[t] @ mus[:, t-1] + B[t] @ us[:,t] + D[t] @ vs[:,t] # + w
        Acl = A[t] - B[t] @ Ps[0][:,:,t] - D[t] @ Ps[1][:,:,t]
        sigmas[t] = Acl  @ sigmas[t-1] @ Acl.T + sigmaW[t]

    return mus, sigmas

if __name__ == "__main__":
    pass
