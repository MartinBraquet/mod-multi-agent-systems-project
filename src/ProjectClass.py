

class Dynamics(object):

    def __init__(self, A, B, D):
        self.Alist = A # list of Ak
        self.Blist = B # list of Bk
        self.Dlist = D # list of Dk
        
class Cost(object):

    def __init__(self, Ru, Rv, muU, muV, sigmaU, sigmaV):
        self.Rulist = Ru
        self.Rvlist = Rv
        
        self.muU = muU
        self.muV = muV
        self.sigmaU = sigmaU
        self.sigmaV = sigmaV

class Project(object):
    def __init__(self, dynamics, cost):
    self.dynamics = dynamics
    self.cost = cost