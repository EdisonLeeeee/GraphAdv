import numpy as np
import scipy.sparse as sp

from graphadv.defense.defender import Defender


def svd_preprocessing(adj, k=50, threshold=0.01, binaryzation=False):
    
    adj = adj.asfptype()
    U, S, V = sp.linalg.svds(adj, k=k)
    adj = (U*S) @ V

    if threshold is not None:
        adj[adj <= threshold] = 0.

    adj = sp.csr_matrix(adj)

    if binaryzation:
        # TODO
        adj.data[adj.data > 0] = 1.0  
        
    return adj


class SVD(Defender):
    def __init__(self, adj, x=None, labels=None, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

    def fit(self, k=50, threshold=0.01, binaryzation=False):
        super().fit()
        self.modified_adj = svd_preprocessing(self.adj, k=k, threshold=threshold, binaryzation=binaryzation)

# TODO
#     @property
#     def edge_flips(self):
#         raise RuntimeError('SVD defense method does not edge flips.')
