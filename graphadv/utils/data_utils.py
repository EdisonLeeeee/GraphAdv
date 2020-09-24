import warnings
import numpy as np
import scipy.sparse as sp


def flip_adj(adj, flips, undirected=True):
    if isinstance(adj, (np.ndarray, np.matrix)):
        if undirected:
            flips = np.vstack([flips, flips[:, [1,0]]])
        return flip_x(adj, flips)
    elif not sp.isspmatrix(adj):
        raise ValueError(f"adj must be a Scipy sparse matrix, but got {type(adj)}.")
        
    if flips is None or len(flips) == 0:
        warnings.warn(
            "There are NO structure flips, the adjacency matrix remain unchanged.",
            RuntimeWarning,
        )
        return adj.tocsr(copy=True)

    rows, cols = np.transpose(flips)
    if undirected:
        rows, cols = np.hstack([rows, cols]), np.hstack([cols, rows])
    data = adj[(rows, cols)].A
    data[data > 0.] = 1.
    data[data < 0.] = 0.
    
    adj = adj.tolil(copy=True)
    adj[(rows, cols)] = 1. - data
    adj = adj.tocsr(copy=False)
    adj.eliminate_zeros()

    return adj

def flip_x(matrix, flips):
    if flips is None or len(flips) == 0:
        warnings.warn(
            "There are NO flips, the matrix remain unchanged.",
            RuntimeWarning,
        )
        return matrix.copy()
    
    matrix = matrix.copy()
    flips = tuple(np.transpose(flips))
    matrix[flips] = 1. - matrix[flips]
    matrix[matrix < 0] = 0
    matrix[matrix > 1] = 1
    return matrix
    
    
def add_edges(adj, edges, undirected=True):
    if isinstance(adj, (np.ndarray, np.matrix)):
        if undirected:
            edges = np.vstack([edges, edges[:, [1,0]]])
        return flip_x(adj, edges)
    elif not sp.isspmatrix(adj):
        raise ValueError(f"adj must be a Scipy sparse matrix, but got {type(adj)}.")
        
    if edges is None or len(edges) == 0:
        warnings.warn(
            "There are NO structure edges, the adjacency matrix remain unchanged.",
            RuntimeWarning,
        )
        return adj.tocsr(copy=True)

    rows, cols = np.transpose(edges)
    if undirected:
        rows, cols = np.hstack([rows, cols]), np.hstack([cols, rows])
    datas = np.ones(rows.size, dtype=adj.dtype)
    
    adj = adj.tocoo(copy=True)
    rows, cols = np.hstack([adj.row, rows]), np.hstack([adj.col, cols])
    datas = np.hstack([adj.data, datas])
    adj = sp.csr_matrix((datas, (rows, cols)), shape=adj.shape)
    adj[adj>1] = 1.    
    adj.eliminate_zeros()
    return adj

def remove_edges(adj, edges, undirected=True):
    if isinstance(adj, (np.ndarray, np.matrix)):
        if undirected:
            edges = np.vstack([edges, edges[:, [1,0]]])
        return flip_x(adj, edges)
    elif not sp.isspmatrix(adj):
        raise ValueError(f"adj must be a Scipy sparse matrix, but got {type(adj)}.")
        
    if edges is None or len(edges) == 0:
        warnings.warn(
            "There are NO structure edges, the adjacency matrix remain unchanged.",
            RuntimeWarning,
        )
        return adj.tocsr(copy=True)

    rows, cols = np.transpose(edges)
    if undirected:
        rows, cols = np.hstack([rows, cols]), np.hstack([cols, rows])
    datas = -np.ones(rows.size, dtype=adj.dtype)
    
    adj = adj.tocoo(copy=True)
    rows, cols = np.hstack([adj.row, rows]), np.hstack([adj.col, cols])
    datas = np.hstack([adj.data, datas])
    adj = sp.csr_matrix((datas, (rows, cols)), shape=adj.shape)
    adj[adj<0] = 0.
    adj.eliminate_zeros()
    return adj
    
