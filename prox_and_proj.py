'''Here we define useful proximity operators and projections'''

import numpy as np

def prox_norm1(x, lam=1): # This function outputs the prox operator of the L1 norm of an n-dim array
    greater, less = x>lam, x<-lam
    y = np.zeros(x.shape)
    y[greater] = x[greater] - lam
    y[less] = x[less] + lam
    return y

def prox_norm1_bis(x, lam=1): # accomplishes the same; same speed
    return (np.abs(x) - lam).clip(min=0) * np.sign(x)

def prox_norm2(x, lam=1):
    if lam == 0: return x
    return ( 1 - lam / np.maximum(np.linalg.norm(x), lam) ) * x

def prox_norm21(x, lam=1, ax=1): # ax is the (only) axis along which the 2-norm is computed
    if lam==0: return x
    if lam==np.inf: return np.zeros(x.shape)
    tup = list(x.shape)
    tup[ax] = 1
    return np.reshape(1 - lam / np.linalg.norm(x, axis=ax).clip(min=lam), tup ) * x

def prox_norminf(x, lam=1): # allows for x n-D; allows for lam=np.inf
    return x - proj_L1_ball(x, lam=lam)

###### PROJECTIONS ######

def proj_L_infty_ball(x, lam=1): # lam is the "radius" of the L infinity ball
    return np.minimum(np.abs(x), lam) * np.sign(x)

def proj_ball_L2(x, lam=1): # lam is the radius of the ball
    return x - prox_norm2(x, lam=lam)

def proj_ball_L_2_infty(x, lam=1):
    return x - prox_norm21(x, lam=lam)

def proj_L1_ball(x, lam=1):
    return proj_simplex( np.abs(x.flatten()), lam=lam ).reshape(x.shape) * np.sign(x)

def proj_L1_ball_2(x, lam=1):
    return proj_simplex_2( np.abs(x.flatten()), lam=lam ).reshape(x.shape) * np.sign(x)

def proj_simplex(x, lam=1): # proj of an array x to the lam-simplex (as if it was 1-D)
    # if lam <= 0: raise ValueError("z must be positive")
    if lam==np.inf: return np.zeros(x.shape)
    x = np.asarray(x).flatten()
    u = np.sort(x)[::-1] # Sort v in descending order
    cssv = np.cumsum(u) # Compute the cumulative sum of u
    rho = np.nonzero(u + (lam - cssv) / np.arange(1, len(u)+1) > 0)[0][-1] # Find the rho value
    theta = (cssv[rho] - lam) / (rho + 1.0) # Compute theta
    return np.maximum(x - theta, 0) # Compute the projection

def prox_square_over_linear(x, t, step, t0=0, t1=np.Inf):
    '''
    prox of ||x||^2 / 2t, where x is (N,d) and t is (N,)
    mask0: identifies elements where t is too small and should be set to t0
    mask1_ identifies elements where t is too large and should be set to t1
    mask: is the inverse of mask0 and mask1; it selects all t values that do not need to be clamped to t0 or t1
    '''
    xsqr = np.sum(x**2, axis=1)

    mask0 = (t <= t0 - (step*xsqr)/(2*(step + t0)**2))
    if t1 < np.Inf:
        mask1 = (t >= t1 - (step*xsqr)/(2*(step + t1)**2))
    else:
        mask1 = np.full(mask0.shape, False)
    mask = ~(mask0 | mask1)

    fac0 = step*xsqr[mask]
    t_mask = np.full(fac0.shape, t0)
    for i in range(20):
        fac1 = 2*(step + t_mask)**3
        t_mask = (fac0*(step + 3*t_mask) + fac1*t[mask])/(2*fac0 + fac1)

        t_prox = np.empty_like(t)
        t_prox[mask0] = t0
        t_prox[mask1] = t1
        t_prox[mask] = t_mask

        x_prox = (t_prox/(step + t_prox))[:,np.newaxis]*x

        return x_prox, t_prox
    

'''This code is a Python translation of the Matlab code by Xiaojing Ye
The original algorithm can be found at http://arxiv.org/abs/1101.6081 or http://ufdc.ufl.edu/IR00000353/'''

# Projection of vector x to the simplex (not used for unbalanced TV-W)
def proj_simplex_2(x, lam=1):
    y = np.asarray(x).flatten()
    n = len(y)
    bget = False
    tmpsum = 0
    s = -np.sort(-y)
    for j in range(n-1):
        tmpsum += s[j]
        tmax = (tmpsum - lam)/(j+1)
        if tmax >= s[j+1]:
            bget = True
            break
    if not(bget): tmax = (tmpsum + s[n-1] - lam)/n
    return (y - tmax).clip(min=0).reshape(x.shape)

''' The following function is my generalization of proj_simplex(y)'''
def proj_simplex_array(A, dim): # projects the last dim axes of an N-dimensional array to the simplex
    shape_flattened = A.shape[:-dim]+(np.prod( A.shape[-dim:] ),) # last d dimensions are flattened
    A_flattened = A.reshape(shape_flattened)
    return np.apply_along_axis(proj_simplex, -1, A_flattened).reshape(A.shape)

