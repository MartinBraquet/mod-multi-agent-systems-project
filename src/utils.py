import numpy as np
from scipy.linalg import sqrtm

def regularizeW(mu_f, Sigma_f, mu_d, Sigma_d):
    """
    This function linearize the Wasserstein distance cost term around the
    mean and covariance of the terminal state of the previous solution. The
    outputs can be used as a quadratic cost function for the LQG game problem.
    Terminal State Cost: x^T Q x + q^T x.

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
    q = (M @ mu_f) - mu_d
    Q = Inx - M
    return Q, q

if __name__ == "__main__":
    pass
