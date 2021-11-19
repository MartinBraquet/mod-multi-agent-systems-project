import numpy as np

from utils import regularizeW

def main():
    # Testing the function
    Sigma_d = np.eye(4)
    mu_d = np.ones((4,1))
    Sigma_f = 2*np.eye(4)
    mu_f = np.zeros((4,1))
    Q, q = regularizeW(mu_f, Sigma_f, mu_d, Sigma_d)
    print(Q, q, sep='\n')

    return 0

if __name__ == "__main__":
    main()
