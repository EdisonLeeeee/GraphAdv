import numpy as np
import scipy.sparse as sp

from graphadv.defense.defender import Defender


class SVD(Defender):
    def __init__(self, adj, x, labels=None, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x, labels=labels, seed=seed, name=name, device=device, **kwargs)

    def fit(self, k=50, threshold=1e-2, binaryzation=False):
        super().fit()

        adj = self.adj.asfptype()
        U, S, V = sp.linalg.svds(adj, k=k)
        diag_S = np.diag(S)
        adj = U @ diag_S @ V

        if threshold is not None:
            adj[adj <= threshold] = 0.

        adj = sp.csr_matrix(adj)

        if binaryzation:
            adj.data = np.clip(adj.data, 0., 1.)

        self.modified_adj = adj

# TODO
#     @property
#     def edge_flips(self):
#         raise RuntimeError('SVD defense method does not edge flips.')
