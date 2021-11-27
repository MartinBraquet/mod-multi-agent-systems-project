import numpy as np

from utils import regularizeW, Wasserstein_Gaussian

def main():
    # Testing the function
    Sigma_d = np.eye(4)
    mu_d = np.zeros((4,1))
    Sigma_f = np.eye(4)
    mu_f = 10 * np.ones((4,1))
    Q, q = regularizeW(mu_f, Sigma_f, mu_d, Sigma_d)
    print(Q, q, sep='\n')
    
    x = mu_f
    dx = x - mu_d
    print(x.T @ Q @ x + 2 * q.T @ x)
    print(Wasserstein_Gaussian(x, Sigma_f, mu_d, Sigma_d))
    
    # Testing the function
    Sigma_d = np.array([[0.05]])
    Sigma_f = Sigma_d
    mu_d = np.array([-1])
    mu_f = np.array([0])
    Q, q = regularizeW(mu_f, Sigma_f, mu_d, Sigma_d)
    print(Q, q, sep='\n')
        
    x = mu_f
    dx = x - mu_d
    print(x.T @ Q @ x + 2 * q.T @ x)
    print(Wasserstein_Gaussian(x, Sigma_f, mu_d, Sigma_d))

    return 0

if __name__ == "__main__":
    main()
