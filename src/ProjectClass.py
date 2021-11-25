class Dynamics(object):
    """
    Alist -> [A_0, A_1, ..., A_{N-1}]
    Blist -> [B_0, B_1, ..., B_{N-1}]
    Dlist -> [D_0, D_1, ..., D_{N-1}]
    dlist -> [d_0, d_1, ..., d_{N-1}]
    sigmaWlist -> [W_0, W_1, ..., W_{N-1}]
    """
    def __init__(self, Alist, Blist, Dlist, dlist, sigmaWlist, mu0, sigma0):
        self.Alist = Alist # list of Ak
        self.Blist = Blist # list of Bk
        self.Dlist = Dlist # list of Dk
        self.dlist = dlist # list of dk
        self.sigmaWlist = sigmaWlist
        self.mu0 = mu0
        self.sigma0 = sigma0

class Cost(object):
    def __init__(self, Ru, Rv, muU, muV, sigmaU, sigmaV, lambda_):
        self.Rulist = Ru
        self.Rvlist = Rv

        self.muU = muU
        self.muV = muV
        self.sigmaU = sigmaU
        self.sigmaV = sigmaV
        self.lambda_ = lambda_

class Project(object):
    def __init__(self, mu0, sigma0, dynamics, cost):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.dynamics = dynamics
        self.cost = cost
