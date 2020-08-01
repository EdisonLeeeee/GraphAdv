import numpy as np
import scipy.sparse as sp
import graphgallery


def is_singleton(adj):
    """Check if the input adjacency matrix has singletons."""
    out_deg = adj.sum(1).A1
    in_deg = adj.sum(0).A1
    return np.where(np.logical_and(in_deg == 0, out_deg == 0))[0].size != 0


def is_self_loops(adj):
    '''Check if the input Scipy sparse adjacency matrix has self loops
    '''
    return adj.diagonal().sum() != 0


def is_binary(x):
    '''Check if the input matrix is unweighted (binary)
    '''
    return np.max(x) == 1. and np.min(x) == 0.


def is_symmetric(adj):
    '''Check if the input Scipy sparse adjacency matrix is symmetric
    '''
    return np.abs(adj - adj.T).sum() == 0


is_scalar_like = graphgallery.is_scalar_like
is_tensor_or_variable = graphgallery.is_tensor_or_variable
