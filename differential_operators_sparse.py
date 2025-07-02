import numpy as np
import scipy.sparse as sp


def create_partialplus_sparse_1d_zeroboundary(n):
    D = np.zeros((n, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(D[:-1,:], -1)   # Set main diagonal to -1
    np.fill_diagonal(D[:, 1:], 1)    # Set upper diagonal to 1
    return sp.csr_matrix( D )

def create_partialplus_sparse_1d_noboundary(n):
    D = np.zeros((n-1, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(D[:,:], -1)   # Set main diagonal to -1
    np.fill_diagonal(D[:, 1:], 1)    # Set upper diagonal to 1
    return sp.csr_matrix( D )

def create_staggered_partialplus_sparse_1d(n):
    D = np.zeros((n-1, n), dtype=int)  # Initialize matrix with zeros
    np.fill_diagonal(D[:,:-1], -1)   # Set main diagonal to -1
    np.fill_diagonal(D[:, 1:], 1)    # Set upper diagonal to 1
    return sp.csr_matrix( D )

def create_partialplus_sparse(shape, i, boundary='zero'):
    '''
    Creates a sparse matrix that computes the partial derivative along the i-th axis of an array with shape 'shape'
    '''
    
    if boundary == 'zero': create_partial_1d = create_partialplus_sparse_1d_zeroboundary
    if boundary == 'no': create_partial_1d = create_partialplus_sparse_1d_noboundary
    if not (-len(shape) <= i < len(shape)):  # Valid range is [-ndim, ndim - 1]
        raise ValueError(f"Invalid axis {i} for shape {shape} with {len(shape)} dimensions")
    i = i % len(shape) # allows for negative indices i (as in numpy's convention)
    
    if len(shape) == 1:
        return create_partial_1d( shape[0] )
    elif i == len(shape)-1:
        result = sp.kron( sp.eye(shape[0], dtype=int) , create_partialplus_sparse(shape[1:], -1, boundary=boundary) )
    else:
        result = sp.kron( create_partialplus_sparse(shape[:-1], i, boundary=boundary) , sp.eye(shape[-1], dtype=int) )
    return result.tocsr()

def create_staggered_partialplus_sparse(shape, i):
    '''
    Creates a sparse matrix that computes the partial derivative along the i-th axis of an array with shape 'shape'
    '''
    if not (-len(shape) <= i < len(shape)):  # Valid range is [-ndim, ndim - 1]
        raise ValueError(f"Invalid axis {i} for shape {shape} with {len(shape)} dimensions")
    i = i % len(shape) # allows for negative indices i (as in numpy's convention)
    
    if len(shape) == 1:
        return create_staggered_partialplus_sparse_1d( shape[0] )
    elif i == len(shape)-1:
        result = sp.kron( sp.eye(shape[0], dtype=int) , create_staggered_partialplus_sparse(shape[1:], -1) )
    else:
        result = sp.kron( create_staggered_partialplus_sparse(shape[:-1], i) , sp.eye(shape[-1], dtype=int) )
    return result.tocsr()

# def nabla_sparse(x, *partials, stack_to_axis=0):
#     return np.stack( [ partial @ x.ravel()  for partial in partials ], axis=stack_to_axis )

def create_nabla_sparse(*partials):
    return sp.bmat( [ [ partial ] for partial in partials ], format='csr' )


def create_laplacian_sparse(*partials):
    return np.sum( [ -partial.T @ partial for partial in partials ] )


# def div_sparse(v, indices, partials_tuple):
#     partial @ v[i] for i, partial in zip(indices, partials_tuple):
#     if v.shape[axis] > v.ndim-1: raise Exception('You want to take too many derivatives')
#     v = np.moveaxis(v, axis, 0)
#     return np.sum( [partialminus(x, k) for k, x in enumerate(v)],  axis=0)

def div(sig, M1, M2): # Computes the divergence of a vector field sig
    return M1 @ sig[:, 0] + M2 @ sig[:, 1]

