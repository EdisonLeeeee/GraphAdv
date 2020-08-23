import warnings
import numpy as np


def flip_adj(adj, flips):
    if flips is None or len(flips) == 0:
        warnings.warn(
            "There are NO structure flips, the adjacency matrix remain unchanged.",
            RuntimeWarning,
        )
        return adj.tocsr(copy=True)

    adj = adj.tolil(copy=True)
    rows, cols = np.transpose(flips)
    rows, cols = np.hstack([rows, cols]), np.hstack([cols, rows])
    data = adj[(rows, cols)].toarray()
    data[data > 0.] = 1.
    adj[(rows, cols)] = 1. - data

    adj = adj.tocsr(copy=False)
    adj.eliminate_zeros()

    return adj


def flip_x(x, flips):
    x = x.copy()
    if flips is None or len(flips) == 0:
        warnings.warn(
            "There are NO feature flips, the feature matrix remain unchanged.",
            RuntimeWarning,
        )    
        return x
    
    for row, col in flips:
        data = 0.0 if x[row, col] > 0. else 1.0
        x[row, col] = data
    return x

    # TODO
    # Using numpy vectorized form to faster
