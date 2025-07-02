import numpy as np
import scipy.sparse as sp

def turn_linear_into_sparse_matrix(f, shape1, *args, **kwargs):
    """
    Given a linear function f acting on numpy arrays of shape `shape1`,
    return a sparse matrix A such that A @ x.flatten() = f(x).flatten().
    """
    size_in = np.prod(shape1)
    x0 = np.zeros(shape1)
    y0 = f(x0, *args, **kwargs)
    shape2 = y0.shape
    size_out = np.prod(shape2)

    A = sp.lil_matrix((size_out, size_in), dtype=np.result_type(y0))

    for j in range(size_in):
        x = np.zeros(shape1)
        np.put(x, j, 1.0)
        y = f(x, *args, **kwargs).flatten()
        A[:, j] = y

    return A.tocsr()

def safe_divide(a, b): # Performs element-wise division a/b, returning 0 where b is 0
    return np.divide(a, b, out=np.zeros_like(a, dtype=b.dtype), where=(b != 0))