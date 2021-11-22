class Dynamics(object):
    def __init__(self, Alist, Blist, Dlist, dlist, sigmaWlist):
        self.Alist = Alist # list of Ak
        self.Blist = Blist # list of Bk
        self.Dlist = Dlist # list of Dk
        self.dlist = dlist # list of dk
        self.sigmaWlist = sigmaWlist

class Cost(object):
    def __init__(self, Ru, Rv, muU, muV, sigmaU, sigmaV):
        self.Rulist = Ru
        self.Rvlist = Rv

        self.muU = muU
        self.muV = muV
        self.sigmaU = sigmaU
        self.sigmaV = sigmaV

class Project(object):
    def __init__(self, mu0, sigma0, dynamics, cost):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.dynamics = dynamics
        self.cost = cost
