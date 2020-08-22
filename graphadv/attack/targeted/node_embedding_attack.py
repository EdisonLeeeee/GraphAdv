import numpy as np
from numba import njit

import scipy.sparse as sp
import scipy.linalg as spl

from graphadv.utils.estimate_utils import (estimate_loss_with_delta_eigenvals,
                                           estimate_loss_with_perturbation_gradient)
from graphadv.utils import filter_singletons
from graphadv.attack.targeted.targeted_attacker import TargetedAttacker


class NodeEmbeddingAttack(TargetedAttacker):
    """ This implementation is not exactly right.
    
    """
    def __init__(self, adj, k=50, name=None, seed=None, **kwargs):
        super().__init__(adj=adj, name=name, seed=seed, **kwargs)
        self.nodes_set = set(range(self.n_nodes))
        
        deg_matrix = sp.diags(self.degree).astype('float64')
        self.vals_org, self.vecs_org = sp.linalg.eigsh(self.adj.astype('float64'), k=k, M=deg_matrix)

    def attack(self, target, n_perturbations=None, dim=32, window_size=5,
               n_neg_samples=3, direct_attack=True, structure_attack=True, feature_attack=False):

        super().attack(target, n_perturbations, direct_attack, structure_attack, feature_attack)
        n_perturbations = self.n_perturbations

        n_nodes = self.n_nodes
        adj = self.adj.astype('float64')

        if direct_attack:
            influencer_nodes = [target]
            candidates = np.column_stack(
                (np.tile(target, n_nodes-1), list(self.nodes_set-set([target]))))
        else:
            influencer_nodes = adj[target].nonzero()[1]
            candidates = np.row_stack([np.column_stack((np.tile(infl, n_nodes - 2),
                                                        list(self.nodes_set - set([target, infl])))) for infl in
                                       influencer_nodes])
        if not self.allow_singleton:
            candidates = filter_singletons(candidates, adj)

        delta_w = 1. - 2 * adj[candidates[:, 0], candidates[:, 1]].A1
        loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w,
                                                                 self.vals_org, self.vecs_org,
                                                                 self.n_nodes,
                                                                 dim, window_size)
        
        self.structure_flips = candidates[loss_for_candidates.argsort()[-n_perturbations:]]
#         print(loss_for_candidates)
#         plt.hist(loss_for_candidates)
