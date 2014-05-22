# R package 'mgcv'-compatible cubic spline basis functions

# These are made available in the patsy.* namespace
__all__ = ["cr", "cs", "cc", "te"]

import numpy as np

from scipy import linalg
from patsy.state import stateful_transform

def get_natural_F(knots):
    """
    Returns matrix F mapping spline values at knots to second derivatives.
    
    knots must be sorted in ascending order
    """
    h = knots[1:]-knots[:-1]
    diag = (h[:-1] + h[1:])/3.
    ul_diag = h[1:-1]/6.
    banded_B = np.array([np.r_[0., ul_diag], diag, np.r_[ul_diag, 0.]])
    D = np.zeros((knots.size - 2, knots.size))
    for i in range(knots.size - 2):
        D[i,i] = 1./h[i]
        D[i,i+2] = 1./h[i+1]
        D[i,i+1] = - D[i,i] - D[i,i+2]
        
    Fm = linalg.solve_banded((1,1), banded_B, D)
    
    return np.vstack([np.zeros(knots.size), Fm, np.zeros(knots.size)])
    

# Cyclic Cubic Regression Splines

def map_cyclic(x, min, max):
    if min >= max:
        raise Exception("Invalid argument: min should be less than max.")
    x[x > max] = min + (x[x > max] - max)%(max - min)
    x[x < min] = max - (min - x[x < min])%(max - min)
    return x

def get_cyclic_F(knots):
    """
    Returns matrix F mapping cyclic spline values at knots to second derivatives.
    
    knots must be sorted in ascending order
    """
    h = knots[1:]-knots[:-1]
    n = knots.size - 1
    B = np.zeros((n,n))
    D = np.zeros((n,n))
    
    B[0,0] = (h[n-1] + h[0])/3.
    B[0,n-1] = h[n-1]/6.
    B[n-1,0] = h[n-1]/6.
    
    D[0,0] = -1./h[0] -1./h[n-1]
    D[0,n-1] = 1./h[n-1]
    D[n-1,0] = 1./h[n-1]
    
    for i in range(1,n):
        B[i,i] = (h[i-1] + h[i])/3.
        B[i,i-1] = h[i-1]/6.
        B[i-1,i] = h[i-1]/6.
    
        D[i,i] = -1./h[i-1] -1./h[i]
        D[i,i-1] = 1./h[i-1]
        D[i-1,i] = 1./h[i-1]
    
    return linalg.solve(B, D)

# Tensor Product

def row_tensor_product(Xs):
    """
    Custom algorithm to precisely match what is done in 'mgcv', in particular look out for order of result columns!
    For reference implementation see 'mat.c', mgcv_tensor_mm(), l.62
    """
    tp_nrows = Xs[0].shape[0]
    tp_ncols = 1
    for X in Xs:
        tp_ncols *= X.shape[1]
    TP = np.zeros((tp_nrows, tp_ncols))
    TP[:,-Xs[-1].shape[1]:] = Xs[-1]
    filled_tp_ncols = Xs[-1].shape[1]
    for X in Xs[-2::-1]:
        p = - filled_tp_ncols * X.shape[1]
        for j in range(X.shape[1]):
            Xj = X[:,j]
            for t in range(-filled_tp_ncols,0):
                TP[:,p] = TP[:,t] * Xj
                p += 1
        filled_tp_ncols *= X.shape[1]
    
    return TP
    
# Common code

def find_knots_lower_bounds(x, knots):
    """
    Returns an array of indices I such that knots[I[i]] < x[i] <= knots[I[i] + 1]
    and I[i] = 0 if x[i] == np.min(knots)
    
    knots must be sorted in ascending order 
    """
    if np.min(x) < np.min(knots) or np.max(x) > np.max(knots):
        raise NotImplementedError("Some data points fall outside the outermost knots.")
        
    lb = np.searchsorted(knots, x) - 1
    lb[lb == -1] = 0
    return lb

def compute_base_functions(x, knots, J = None):
    if J == None:
        J = find_knots_lower_bounds(x, knots)
    h = knots[1:]-knots[:-1]
    hj = h[J]
    xj1_x = knots[J+1] - x
    x_xj = x - knots[J]
    
    return xj1_x/hj, x_xj/hj, (xj1_x*(xj1_x*xj1_x/hj - hj))/6., (x_xj*(x_xj*x_xj/hj - hj))/6.

def apply_constraints(X, Cp):
    """
    Applies the parameters constraints given by the matrix Cp to the free design matrix X.
    """
    m = Cp.shape[0]
    Q, R = linalg.qr(np.transpose(Cp))
    
    return np.dot(X, Q[:,m:])

def get_free_crs_dmatrix(x, knots, cyclic = False):
    """
    Returns prediction matrix with dimensions len(x) x n
    for a cubic regression spline smoother
    where 
      n = len(knots)       for natural CRS  
      n = len(knots) - 1   for cyclic CRS
    
    knots must be sorted in ascending order
    """
    n = knots.size
    if cyclic:
        x = map_cyclic(x, min(knots), max(knots))
        n = n - 1
    elif np.min(x) < np.min(knots) or np.max(x) > np.max(knots):
        raise NotImplementedError("Natural cubic regression spline: some data points fall outside the outermost knots.")
        
    J = find_knots_lower_bounds(x, knots)
    J1 = J + 1
    if cyclic:
        J1[J1==n] = 0
    
    Id = np.identity(n)
    
    if cyclic:
        F = get_cyclic_F(knots)
    else:
        F = get_natural_F(knots)
    
    ajm, ajp, cjm, cjp = compute_base_functions(x, knots, J)
    XT = ajm * Id[J,:].T + ajp * Id[J1,:].T + cjm * F[J,:].T + cjp * F[J1,:].T
        
    return XT.T

def get_crs_dmatrix(x, knots, Cp = None, cyclic = False):
    """
    Returns prediction matrix with dimensions len(x) x n
    where:
        n = len(knots) - nrows(Cp)       for natural CRS  
        n = len(knots) - nrows(Cp) - 1   for cyclic CRS
    for a cubic regression spline smoother
    
    knots must be sorted in ascending order
    Cp is the parameters constraints matrix (C\beta = 0)
    """
    X = get_free_crs_dmatrix(x, knots, cyclic)
    if Cp is not None:
        X = apply_constraints(X, Cp)
    
    return X

def get_te_dmatrix(Xs, Cp = None):
    """
    Returns tensor product design matrix of given smooths design matrices 
    
    Cp is the parameters constraints matrix (Cp\beta = 0) 
    """
    X = row_tensor_product(Xs)
    if Cp is not None:
        X = apply_constraints(X, Cp)
    
    return X

class CR(object):
    """
    cr(x, knots, parameters_constraints)
    cs(x, knots, parameters_constraints)
    """
    def __init__(self):
        self._xs = []
        self._knots = None
        self._parameters_constraints = None
    
    def memorize_chunk(self, x, knots, parameters_constraints=None):
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError("Input to 'cs' must be 1-d, or a 2-d column vector.")
        self._xs.append(x)
        self._knots = knots
        self._parameters_constraints = parameters_constraints

    def memorize_finish(self):
        xs = np.concatenate(self._xs)
        #TODO: use 'xs' to compute knots and identifiability constraint
        pass

    def transform(self, x, knots, parameters_constraints=None):
        return get_crs_dmatrix(x, knots, self._parameters_constraints)
    
cr = stateful_transform(CR)
cs = stateful_transform(CR)

class CC(object):
    """
    cc(x, knots, parameters_constraints)
    """
    def __init__(self):
        self._xs = []
        self._knots = None
        self._parameters_constraints = None
    
    def memorize_chunk(self, x, knots, parameters_constraints=None):
        x = np.atleast_1d(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim > 1:
            raise ValueError("Input to 'cc' must be 1-d, or a 2-d column vector.")
        self._xs.append(x)
        self._knots = knots
        self._parameters_constraints = parameters_constraints

    def memorize_finish(self):
        xs = np.concatenate(self._xs)
        #TODO: use 'xs' to compute knots and identifiability constraint
        pass

    def transform(self, x, knots, parameters_constraints=None):
        return get_crs_dmatrix(x, knots, self._parameters_constraints, cyclic=True)
    
cc = stateful_transform(CC)

class TE(object):
    """
    te(*args, **kwargs)
    """
    def __init__(self):
        pass
    
    def memorize_chunk(self, *args, **kwargs):
        pass

    def memorize_finish(self):
        pass

    def transform(self, *args, **kwargs):
        return get_te_dmatrix(args, kwargs.get('parameters_constraints', None))
    
te = stateful_transform(TE) # We don't really need a stateful transform here 
