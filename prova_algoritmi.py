import numpy as np

def hilbert_matrix(n, dim=2): # creates a dim_dimensional Hilbert matrix (ill-conditioned)
    grids = np.ogrid[tuple(slice(0, n) for _ in range(dim))] # Create a grid of indices
    # Sum the indices element-wise and add 1 to avoid division by zero
    total = sum(grids) + 1
    return 1.0 / total


def prova_CP(A, C, b, iterations=10000): # min_{Cx >= 0} = 1/2 ||Ax - b||^2
    
    d = A.shape[-1]
    if A.shape[0] != len(b): print('b dim does not coincide with the output of A')
    
    import cvxpy as cp
    x = cp.Variable(d)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [C @ x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print(problem.status)
    x_cp = x.value
    
    I = np.eye(d)
    
    def prox_f(x, tau):
        return { key: np.linalg.inv( I + tau * A.T @ A ) @ ( x[key] + tau * A.T @ b ) for key in x }
    
    def prox_gstar(u, sigma):
        return { key: u[key].clip(max=0) for key in u }
    
    # Identity operator
    def L(x):
        return {'u': C @ x['x']}
    
    def Lstar(u):
        return {'x': C.T @ u['u']}
    
    L_norm = max(np.linalg.svd(C, compute_uv=False))
    # L_norm_sq = np.linalg.norm(C)**2
    param_factor = 3000 # for best convergence tau is 10^3 times sigma!
    tau = param_factor / L_norm
    sigma = 1 / ( param_factor * L_norm)
    
    # Initialize variables
    x0 = {'x': np.zeros(d)}  # Starting at zero
    u0 = L(x0)  # Dual variable initialized to zero

    # Run Chambolle-Pock Algorithm
    from algorithms_general import CP
    x_opt, u_opt = CP(x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=True)
    x_opt = x_opt['x']
    
    return x_opt, x_cp

def prova_pDR_Richardson(A, C, b, iterations=10000): # min_{Cx >= 0} = 1/2 ||Ax - b||^2; confronto con soluzione di scipy
    
    import cvxpy as cp
    
    n = A.shape[1]
    
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [C @ x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    x_cp = x.value
        
    I = np.eye(n)
    
    def prox_f(x, tau):
        return { 'x': np.linalg.inv( I + tau * A.T @ A ) @ ( x['x'] + tau * A.T @ b ) }
    
    def prox_g(y, sigma):
        return { 'y': y['y'].clip(max=0) }
    
    def K(x):
        return { 'y': C @ x['x'] }
    def Kstar(y):
        return { 'x': C.T @ y['y'] }
    
    # Initialize variables
    x0 = {'x': np.zeros(n)}  # Starting at zero

    # Run Chambolle-Pock Algorithm
    from algorithms_general import pDR_Richardson
    
    K_norm_sq = max(np.linalg.svd(C, compute_uv=False))**2
    # K_norm_sq = np.sum( C**2 )
    
    sig=1
    print(sig)
    
    x_opt, y_opt = pDR_Richardson(x0, sig, prox_f, prox_g, K, Kstar, K_norm_sq, iterations=iterations)
    x_opt = x_opt['x']
    
    return x_opt, x_cp

m = 8
A = hilbert_matrix(m)
# A = np.random.rand(m,m)
np.random.seed(0)
C = hilbert_matrix(m) + np.random.rand(m,m)/m
b = np.random.rand(m)

iterations = 10000
x_opt_CP, x_cvxpy = prova_CP(A, C, b, iterations=iterations)
# x_opt_DR, _ = prova_pDR_Richardson(A, C, b, iterations=iterations)

from default import printarr
printarr(x_cvxpy, decimals=4)
printarr(x_opt_CP, decimals=4)
print(np.min(C @ x_cvxpy))
print(np.min(C @ x_opt_CP))
print('||A @ x_cvxpy - b||^2 = ', np.linalg.norm(A @ x_cvxpy - b)**2)
print('||A @ x_opt_CP - b||^2 = ', np.linalg.norm(A @ x_opt_CP - b)**2)
# printarr(C @ x_opt_DR)np.linalg.norm(A @ x_opt_CP - b))
print('||x_opt_CP - x_cvxpy|| = ', np.linalg.norm(x_opt_CP - x_cvxpy))
# printarr(C @ x_opt_DR)
# print(np.linalg.norm(A @ x_opt_DR - b))

def prova_DR(A, b): # min_{x >= 0} = 1/2 ||Ax - b||^2; confronto con soluzione di scipy

    from scipy.optimize import nnls    
    x_scipy, residual = nnls(A, b)
    
    I = np.eye(A.shape[1])
    
    def prox_f(x, tau):
        return { 'x': np.linalg.inv( I + tau * A.T @ A ) @ ( x['x'] + tau * A.T @ b ) }
    
    # Proximal operator of g*(u) = 0 (identity function)
    def prox_g(s, sigma):
        return {'x': s['x'].clip(min=0) }
    
    # Initialize variables
    s0 = {'x': np.zeros(A.shape[1])}  # Starting at zero

    # Run Chambolle-Pock Algorithm
    from algorithms_general import DR
    
    x_opt, s_opt = DR(s0, prox_f, prox_g, iterations=10000 )
    
    return x_opt, x_scipy
