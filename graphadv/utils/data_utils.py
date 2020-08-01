import warnings
import numpy as np


def flip_adj(adj, flips):
    if len(flips) == 0:
        warnings.warn(
            "No flips.",
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
    for row, col in flips:
        data = 1.0 if x[row, col] > 0. else 0.0
        x[row, col] = 1. - data

    # TODO
    # Using numpy vectorized form to faster
