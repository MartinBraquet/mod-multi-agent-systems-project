import numpy as np

from utils import regularizeW, Wasserstein_Gaussian

def main():
    # Testing the function
    Sigma_d = np.eye(4)
    mu_d = np.zeros((4,1))
    Sigma_f = 4 * np.eye(4)
    mu_f = 10 * np.ones((4,1))
    Q, q = regularizeW(mu_f, Sigma_f, mu_d, Sigma_d)
    print(Q, q, sep='\n')
        
    print(Wasserstein_Gaussian(mu_f, Sigma_f, mu_d, Sigma_d))
    
    x = mu_f
    print(x.T @ Q @ x + q.T @ x)

    return 0

if __name__ == "__main__":
    main()