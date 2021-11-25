import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
from ProjectClass import Dynamics, Cost

def getMatirces(Dynamics, Cost):
    """
    This functions takes the dynamics class as input and outputs the required
    matrices and cvxpy.variables to turn the covariance steering problem into a
    finite dimensional optimization problem.
    """
    Alist = Dynamics.Alist
    Blist = Dynamics.Blist
    Dlist = Dynamics.Dlist
    zlist = Dynamics.zlist
    sigmaWlist = Dynamics.sigmaWlist

    Rulist = Cost.Rulist
    Rvlist = Cost.Rvlist

    N = len(Alist)  # Problem horizon
    nx, nu, nv = Alist[0].shape[1], Blist[0].shape[1], Dlist[0].shape[1]

    # Set Constant Matirces:
    Gamma = []
    for i in range(N+1):
        Gamma.append(Phi_func(Alist, i, 0))
    Gamma = cp.vstack(Gamma)

    block_Hu, block_Hv, block_Hw = [], [], []
    for i in range(N+1):
        row_Hu, row_Hv, row_Hw = [], [], []
        for j in range(N):
            if j < i:
                row_Hu.append(Phi_func(Alist, i, j) @ Blist[j])
                row_Hv.append(Phi_func(Alist, i, j) @ Dlist[j])
                row_Hw.append(Phi_func(Alist, i, j))
        block_Hu.append(np.hstack(row_Hu))
        block_Hv.append(np.hstack(row_Hv))
        block_Hw.append(np.hstack(row_Hw))
    Hu, Hv, Hw = np.vstack(block_Hu), np.vstack(block_Hv), np.vstack(block_Hw)

    Z = np.hstack(zlist)

    Wbig = np.zeros((nx*N, nx*N))
    for k in range(N):
        Wbig[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = sigmaWlist[k]

    Rubig = np.zeros((nu*N, nu*N))
    Rvbig = np.zeros((nv*N, nv*N))
    for k in range(N):
        Rubig[i*nu:(i+1)*nu, i*nu:(i+1)*nu] = Rulist[k]
        Rvlist[i*nv:(i+1)*nv, i*nv:(i+1)*nv] = Rvlist[k]

    return Gamma, Hu, Hv, Hw, Z, Wbig, Rubig, Rvbig

# def getObjectiveFunction():
#     """
#     This function takes the Cost class along with a solution of covariance
#     steering problem and returns a cvxpy expression which represents the
#     """
#     return

def setDecisionVariables(Dynamics, **init_values):
    """
    This function creates the necessary decision variables given the system
    dynamics.
    """
    nx = Dynamics.Alist[0].shape[1]
    nu = Dynamics.Blist[0].shape[1]
    nv = Dynamics.Dlist[0].shape[1]
    N = len(Alist)

    # Extract Initial Values:
    for key, value in init_values.items():
        if key == "Ubar":
            Ubar_init = value
        elif key == "Vbar":
            Vbar_init = value
        elif key == "Lu":
            Lu_init = value
        elif key == "Lv":
            Lv_init = value
        elif key == "Ku":
            Ku_init = value
        elif key == "Kv":
            Kv_init = value
        else:
            print("Unknown Keyword for initial values for decision variables")

    if "Ubar" not in init_values.keys():
        Ubar_init = np.zeros((nu*N, 1))
    if "Vbar" not in init_values.keys():
        Vbar_init = np.zeros((nv*N, 1))
    if "Lu" not in init_values.keys():
        Lu_init = np.zeros((nu*N, nx))
    if "Lv" not in init_values.keys():
        Lv_init = np.zeros((nv*N, 1))
    if "Ku" not in init_values.keys():
        Ku_init = np.zeros((nu*N, nx*(N+1)))
    if "Kv" not in init_values.keys():
        Kv_init = np.zeros((nv*N, nx*(N+1)))

    # Set decision variables:
    Ubar = cp.Variable((nu*N, 1), value=Ubar_init)
    Vbar =  cp.Variable((nv*N, 1), value=Vbar_init)
    Lu = cp.Variable((nu*N, nx), value=Lu_init),
    Lv = cp.Variable((nv*N, nx), value=Lv_init)
    block_Ku, block_Kv = [], []
    for i in range(N):
        row_Ku = []
        row_Kv = []
        for j in range(N):
            Ku_init_ij = Ku_init[i*nu:(i+1)*nu, j*nx:(j+1)*nx]
            Kv_init_ij = Kv_init[i*nu:(i+1)*nu, j*nx:(j+1)*nx]
            if j < i:
                row_Ku.append(cp.Variable((nu, nx)))
                row_Kv.append(cp.Variable((nv, nx)))
            else:
                row_Ku.append(np.zeros((nu, nx)))
                row_Kv.append(np.zeros((nv, nx)))

        block_Ku.append(cp.hstack(row_Ku))
        block_Kv.append(cp.hstack(row_Kv))

    Ku, Kv = cp.vstack(block_Ku), cp.vstack(block_Kv)

    return Ubar, Vbar, Lu, Lv, Ku, Kv

def Phi_func(Alist, k2, k1):
    """
    State Transition function.
    Alist -> [A_0, A_1, ..., A_{N-1}]
    k2 -> final time step
    k1 -> initial time step.
    Phi_func(Alist, k, k) = np.eye(nx)
    """
    nx = Alist[0].shape[1]
    assert k2 >= k1

    Phi_matrix = np.eye(nx)
    for k in range(k1, k2):
        Phi_matrix = Alist[k] @ Phi_matrix

    return Phi_matrix

def solveCCPforU(Dynamics, Cost, Upolicy, Vpolicy,
                 CCP_iter=10, solver="MOSEK", eps=1e-4):
    """
    Inputs:
        Dynamics -> System Dynamics Class
        Cost -> Covariance Steering Cost Class
        Upolicy -> (Ubar_value, Lu_value, Ku_value) value of the expressions
            of PlayerU's policy from the previous iteration
        Vpolicy -> (Vbar_value, Lv_value, Kv_value) value of the expressions
            of PlayerV's policy from the previous iteration
        CCP_iter -> number of CCP iterations
        solver -> Convex Solver for CCP subproblems
    Outputs:
        Success -> flag denotes the convergence
    """
    Alist = Dynamics.Alist # list of Ak
    Blist = Dynamics.Blist # list of Bk
    Dlist = Dynamics.Dlist # list of Dk
    dlist = Dynamics.dlist # list of dk
    sigmaWlist = Dynamics.sigmaWlist
    mu0 = Dynamics.mu0
    sigma0 = Dynamics.sigma0

    Rulist = Cost.Ru
    Rvlist = Cost.Rv

    muU = Cost.muU
    # muV = Cost.muV
    sigmaU = Cost.sigmaU
    # sigmaV = Cost.sigmaV
    lambda_ = Cost.lambda_

    Ubar_init, Lu_init, Ku_init = Upolicy
    Vbar_init, Lv_init, Kv_init = Vpolicy

    Gamma, Hu, Hv, Hw, Z, Wbig, Rubig, Rvbig = getMatirces(Dynamics, Cost)

    Ubar, _, Lu, _, Ku, _ = setDecisionVariables(Dynamics,
                                        Ubar=Ubar_init,
                                        Lu=Lu_init,
                                        Ku=Ku_init,
                                        Vbar=Vbar_init,
                                        Lv=Lv_init
                                        Kv=Kv_init)


    u, s, v = svd(Rubig)
    sqrtRubig = u @ np.diag(np.sqrt(s)) @ v
    u, s, v = svd(sigma0)
    sqrtSigma0 = u @ np.diag(np.sqrt(s)) @ v
    u, s, v = svd(Wbig)
    sqrtWbig = u @ np.diag(np.sqrt(s)) @ v
    u, s, v = svd(sigmaU)
    sqrtSigmaU = u @ np.diag(np.sqrt(s)) @ v

    Pf = np.zeros((nx, nx*(N+1)))
    Pf[:,-nx:] = np.eye(nx)

    control_cost = (cp.norm(Ubar ,2)**2
                    + cp.norm(sqrtRubig@Lu@sqrtSigma0, "fro")**2
                    + cp.norm(sqrtRubig@Ku@sqrtWbig, "fro")**2)

    Vbar_value, Lv_value, Kv_value = Vbar_init, Lu_init, Kv_init
    zeta = (Pf @
        (cp.hstack([Gamma + Hu@Lu + Hv@Lv_value, Hw + Hu@Ku + Hv@Kv_value])
                @ np.block([ [sqrtSigma0, np.zeros(nx, nx*N)],
                             [np.zeros(nx*N, nx), sqrtWbig] ]) )
                             )
    mu_f = Pf @ (Gamma @ mu0 + Hw @ Z + Hu @ Ubar + Hv @ Vbar_value)

    cvx_wass_cost = (cp.norm(mu_f-muU,2)**2 + cp.norm(zeta,"fro")**2
                        + np.trace(sigmaU))
    ccv_wass_cost = -2 * cp.norm( sqrtSigmaU @ zeta ,"nuc")

    total_cost = control_cost + lambda_ * (cvx_wass_cost + ccv_wass_cost)
    obj_value_prev = total_cost.value
    convergence_flag = False
    for i in range(CCP_iter):
        zeta0 = zeta.value
        helper = np.linalg.inv(sqrtm(sqrtSigmaU@zeta0 @ zeta0.T@sqrtSigmaU))
        concave_grad = sqrtSigmaU @ helper @ sqrtSigmaU @ zeta0

        cvx_subproblem_cost = (control_cost
                        + lambda_*(cvx_wass_cost
                        + ccv_wass_cost.value
                        - 2*cp.trace(concave_grad.T @ (zeta - zeta0)  ) )
                                )
        subproblem_obj = cp.Minimize(cvx_subproblem_cost)
        subprob = cp.Problem(subproblem_obj)
        subprob.solve(solver=solver)
        obj_value_new = total_cost.value
        if np.abs(obj_value_new-obj_value_prev) <= eps:
            convergence_flag = True
            break

    Upolicy_value = Ubar.value, Lu.value, Ku.value

    control_cost_v = (np.norm(Vbar ,2)**2
                    + np.norm(sqrtRvbig@Lv_value@sqrtSigma0, "fro")**2
                    + np.norm(sqrtRvbig@Kv_value@sqrtWbig, "fro")**2)

    wass_cost_v = (np.norm(mu_f-muV,2)**2 + np.norm(zeta.value,"fro")**2
                        + np.trace(sigmaV)
                        - 2 * np.norm( sqrtSigmaV @ zeta.value ,"nuc"))

    total_cost_v = control_cost_v + lambda_ * wass_cost_v
    
    return Upolicy_value, convergence_flag, total_cost.value, total_cost_v

def solveCCPforV(Dynamics, Cost, Upolicy, Vpolicy,
                 CCP_iter=10, solver="MOSEK", eps=1e-4):
    """
    Inputs:
        Dynamics -> System Dynamics Class
        Cost -> Covariance Steering Cost Class
        Upolicy -> (Ubar, Lu, Ku) where each term is a cvxpy expression
        VpolicyPrev -> (Vbar_value, Lv_value, Kv_value) value of the expressions
            of PlayerV's policy from the previous iteration
        CCP_iter -> number of CCP iterations
        solver -> Convex Solver for CCP subproblems
    Outputs:
        Success -> flag denotes the convergence
    """
    Alist = Dynamics.Alist # list of Ak
    Blist = Dynamics.Blist # list of Bk
    Dlist = Dynamics.Dlist # list of Dk
    dlist = Dynamics.dlist # list of dk
    sigmaWlist = Dynamics.sigmaWlist
    mu0 = Dynamics.mu0
    sigma0 = Dynamics.sigma0

    Rulist = Cost.Ru
    Rvlist = Cost.Rv

    muU = Cost.muU
    muV = Cost.muV
    sigmaU = Cost.sigmaU
    sigmaV = Cost.sigmaV
    lambda_ = Cost.lambda_

    Ubar_init, Lu_init, Ku_init = Upolicy
    Vbar_init, Lv_init, Kv_init = Vpolicy

    Gamma, Hu, Hv, Hw, Z, Wbig, Rubig, Rvbig = getMatirces(Dynamics, Cost)

    _, Vbar, _, Lv, _, Kv = setDecisionVariables(Dynamics,
                                        Ubar=Ubar_init,
                                        Lu=Lu_init,
                                        Ku=Ku_init,
                                        Vbar=Vbar_init,
                                        Lv=Lv_init
                                        Kv=Kv_init)


    u, s, v = svd(Rvbig)
    sqrtRvbig = u @ np.diag(np.sqrt(s)) @ v
    u, s, v = svd(sigma0)
    sqrtSigma0 = u @ np.diag(np.sqrt(s)) @ v
    u, s, v = svd(Wbig)
    sqrtWbig = u @ np.diag(np.sqrt(s)) @ v
    u, s, v = svd(sigmaV)
    sqrtSigmaV = u @ np.diag(np.sqrt(s)) @ v

    Pf = np.zeros((nx, nx*(N+1)))
    Pf[:,-nx:] = np.eye(nx)

    control_cost = (cp.norm(Vbar ,2)**2
                    + cp.norm(sqrtRvbig@Lv@sqrtSigma0, "fro")**2
                    + cp.norm(sqrtRvbig@Kv@sqrtWbig, "fro")**2)

    Ubar_value, Lu_value, Ku_value = Ubar_init, Lu_init, Kv_init
    zeta = (Pf @
        (cp.hstack([Gamma + Hu@Lu_value + Hv@Lv, Hw + Hu@Ku_value + Hv@Kv])
                @ np.block([ [sqrtSigma0, np.zeros(nx, nx*N)],
                             [np.zeros(nx*N, nx), sqrtWbig] ]) )
                             )
    mu_f = Pf @ (Gamma @ mu0 + Hw @ Z + Hu @ Ubar_value + Hv @ Vbar)

    cvx_wass_cost = (cp.norm(mu_f-muV,2)**2 + cp.norm(zeta,"fro")
                        + np.trace(sigmaV))
    ccv_wass_cost = -2 * cp.norm(sqrtSigmaV @ zeta ,"nuc")

    total_cost = control_cost + lambda_ * (cvx_wass_cost + ccv_wass_cost)
    obj_value_prev = total_cost.value
    for i in range(CCP_iter):
        zeta0 = zeta.value
        helper = np.linalg.inv(sqrtm(sqrtSigmaV@zeta0 @ zeta0.T@sqrtSigmaU))
        concave_grad = sqrtSigmaV @ helper @ sqrtSigmaV @ zeta0

        cvx_subproblem_cost = (control_cost
                        + lambda_*(cvx_wass_cost
                        + ccv_wass_cost.value
                        - 2*cp.trace(concave_grad.T @ (zeta - zeta0)  ) )
                                )
        subproblem_obj = cp.Minimize(cvx_subproblem_cost)
        subprob = cp.Problem(subproblem_obj)
        subprob.solve(solver=solver)
        obj_value_new = total_cost.value
        if np.abs(obj_value_new-obj_value_prev) <= eps:
            break

    Vpolicy_value = Vbar.value, Lv.value, Kv.value

    return Vpolicy.value

def solveCCPboth():
    return 0
