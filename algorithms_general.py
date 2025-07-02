'''Here we define a general version of Chambolle-Pock, using dictionary variables'''
'''I refer to Condat (https://doi.org/10.1137/20M1379344) for the notation. Here, the saddle problem has F and G^*'''

import numpy as np
import time

def CP(x0, u0, tau, sigma, prox_f, prox_gstar, L, Lstar, iterations, printprogress=False): # Condat Notation (f+g)
    """
    Chambolle-Pock (CP) algorithm as seen in Condat's article, with in-memory checkpointing.
    Saves state in the `current_state` variable on KeyboardInterrupt.
    """
    x = x0.copy()
    u = u0.copy()
    start_time = time.time()
    next_update = start_time  # Update every 1 second

    for i in range(1, iterations + 1):
        x_prev = x.copy()        
        # x_{i+1} = prox_{tau f}( x_i - tau * Lstar(u_i) )
        x_minus = subtract_dicts(x, scalar_multiply_dict(Lstar(u), tau))
        x = prox_f(x_minus, tau)
        # u_{i+1} = prox_{sigma g^*}( u_i + sigma * L( 2 * x_{i+1} - x_i ) )
        x_tilde = subtract_dicts(scalar_multiply_dict(x, 2), x_prev)
        u_plus = add_dicts(u, scalar_multiply_dict(L(x_tilde), sigma))
        u = prox_gstar(u_plus, sigma)
        
        current_time = time.time()
        if (current_time >= next_update) & printprogress:
            print(f"Progress: ~{int(100 * i / iterations)}% ({i}/{iterations}),  Time elapsed:\
~{int((current_time - start_time) / 60)} min,  Time left:\
~{int((current_time - start_time) / 60 * (iterations - i) / i)} min")
            next_update = current_time + 30  # Schedule next update
    return x, u


def pDR_Richardson(x0, sig, prox_f, prox_g, K, Kstar, norm_K_sq, iterations=10000, printprogress=False):
    # Bredies notation (f+g*)
    """
    Preconditioned Douglas-Rachford (PDR) algorithm.
    
    Inputs:
        x0, xbar0, y0, ybar0: Initial primal and dual variables (dicts of arrays).
        sigma: Step-size parameter.
        prox_f, prox_g: Proximal operators for F and G (functions taking (dict, sigma)).
        K, Kstar: Forward and adjoint linear operators (functions taking dicts).
        M_inv: Inverse of preconditioner M (function taking a dict).
        T: Function computing T(x) = x + sigma^2 * K^T K x (returns dict).
        iterations: Number of iterations.
        printprogress: If True, prints progress every ~30 seconds.
    """
    x = x0.copy()
    xbar = x.copy()
    y = K(x)
    ybar = y.copy()

    
    T = lambda x: add_dicts( x, scalar_multiply_dict( Kstar( K(x) ), sig**2 ) )
    lam = 1 + sig**2 * norm_K_sq
    M_inv = lambda x: scalar_multiply_dict(x, 1/lam)
    
    start_time = time.time()
    next_update = start_time
    
    # M_inv is initialized at lam=1 as in the Richardson preconditioner

    for i in range(1, iterations + 1):
        # b^k = x̄^k - σ K^T ȳ^k
        b = subtract_dicts(xbar, scalar_multiply_dict(Kstar(ybar), sig))
        # x^{k+1} = x^k + M^{-1}( b^k - T(x^k) )
        diff = subtract_dicts(b, T(x))
        x = add_dicts(x, M_inv(diff))
        # y^{k+1} = ȳ^k + σ K x^{k+1}
        y = add_dicts(ybar, scalar_multiply_dict(K(x), sig))
        # x̄^{k+1} = x̄^k + prox_{σF}( 2x^{k+1} - x̄^k ) - x^{k+1}
        two_x = scalar_multiply_dict(x, 2)
        prox_arg = subtract_dicts(two_x, xbar)
        prox_x = prox_f(prox_arg, sig)
        xbar = add_dicts(xbar, subtract_dicts(prox_x, x))
        # ȳ^{k+1} = ȳ^k + prox_{σG}( 2y^{k+1} - ȳ^k ) - y^{k+1}
        two_y = scalar_multiply_dict(y, 2)
        prox_arg = subtract_dicts(two_y, ybar)
        prox_y = prox_g(prox_arg, sig)
        ybar = add_dicts(ybar, subtract_dicts(prox_y, y))
        current_time = time.time()
        if (current_time >= next_update) and printprogress:
            elapsed = int((current_time - start_time) / 60)
            remaining = int(elapsed * (iterations - i) / i) if i > 0 else '?'
            print(f"Progress: ~{int(100 * i / iterations)}% ({i}/{iterations}),  "
                  f"Time elapsed: ~{elapsed} min,  Time left: ~{remaining} min")
            next_update = current_time + 30

    return x, y

def DR(s0, prox_f, prox_g, iterations, tau=1, rho=lambda i: 1, printprogress=False): # Condat notation (f+g)
    """
    Douglas-Rachford (DR) splitting algorithm using dictionaries.
    
    Parameters:
    - s0: Initial variable (dictionary)
    - tau: Step size
    - prox_f: Proximal operator of f
    - prox_g: Proximal operator of g
    - iterations: Number of iterations
    - rho: Relaxation parameter (default=1). Note: it's a function!
    - printprogress: If True, prints progress updates
    
    Returns:
    - Final solution s
    """
    s = s0.copy()
    start_time = time.time()
    next_update = start_time  # Time update interval
    
    for i in range(1, iterations + 1):
        s_prev = s.copy()

        # First proximal step: x_half = prox_f(s)
        x_half = prox_f(s, tau)

        # Reflection step: compute x_tilde = 2x_half - s
        x_tilde = subtract_dicts(scalar_multiply_dict(x_half, 2), s)

        # Second proximal step: prox_g(x_tilde)
        prox_x_tilde = prox_g(x_tilde, tau)

        # Final update step
        s = add_dicts(s_prev, scalar_multiply_dict(subtract_dicts(prox_x_tilde, x_half), rho(i)))
        
        # Progress logging
        current_time = time.time()
        if (current_time >= next_update) & printprogress:
            print(f"Progress: ~{int(100 * i / iterations)}% ({i}/{iterations}), Time elapsed:\
~{int((current_time - start_time) / 60)} min, Time left:\
~{int((current_time - start_time) / 60 * (iterations - i) / i)} min")
            next_update = current_time + 30  # Schedule next update

    return x_half, s



def add_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys(): raise Exception('The keys don''t match')
    return {key: dict1[key] + dict2[key] for key in dict1}

def subtract_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys(): raise Exception('The keys don''t match')
    return {key: dict1[key] - dict2[key] for key in dict1}

def scalar_multiply_dict(dict0, scalar):
    return {key: scalar * dict0[key] for key in dict0}

def inner_product_dicts(dict1, dict2): # useful to verify that the adjoint operator is correctly defined
    return np.sum( [ np.dot( dict1[key].flatten(), dict2[key].flatten() ) for key in dict1 ] )

