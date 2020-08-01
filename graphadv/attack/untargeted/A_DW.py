import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl

from numba import njit
from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphadv.utils.estimate_utils import (estimate_loss_with_delta_eigenvals,
                                           estimate_loss_with_perturbation_gradient)
from graphadv.utils.graph_utils import edges_to_sparse


class A_DW(UntargetedAttacker):
    def __init__(self, adj, name=None, seed=None, **kwargs):
        super().__init__(adj=adj, name=name, seed=seed, **kwargs)

    def attack(self, n_perturbations=0.05, dim=32, window_size=5, k=100,
               addition=False, removel=True, structure_attack=True, feature_attack=False):

        if not (addition or removel):
            raise RuntimeError('Edge addition and removel cannot be conducted simultaneously.')

        super().attack(n_perturbations, structure_attack, feature_attack)
        n_perturbations = self.n_perturbations

        adj = self.adj.astype('float64')

        candidates = []
        if addition:
            n_candidates = min(2*n_perturbations, self.n_edges)
            candidate = self.generate_candidates_addition(adj, n_candidates)
            candidates.append(candidate)

        if removel:
            candidate = self.generate_candidates_removal(adj)
            candidates.append(candidate)

        candidates = np.vstack(candidates)

        delta_w = 1. - 2 * adj[candidates[:, 0], candidates[:, 1]].A1

        # generalized eigenvalues/eigenvectors
        # whether to use sparse formate
        if k is None:
            deg_matrix = np.diag(self.degree).astype('float64', copy=False)
            vals_org, vecs_org = spl.eigh(adj.toarray(), deg_matrix)
        else:
            deg_matrix = sp.diags(self.degree).astype('float64', copy=False)
            vals_org, vecs_org = sp.linalg.eigsh(adj, k=k, M=deg_matrix)

        loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w,
                                                                 vals_org, vecs_org,
                                                                 self.n_nodes,
                                                                 dim, window_size)
        
        self.dim = dim
        self.structure_flips = candidates[loss_for_candidates.argsort()[-n_perturbations:]]
        self.window_size = window_size

    def generate_candidates_removal(self, adj):
        """Generates candidate edge flips for removal (edge -> non-edge),
         disallowing one random edge per node to prevent singleton nodes.

        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        n_nodes = self.n_nodes
        deg = np.where(self.degree == 1)[0]
        adj = self.adj

        hiddeen = np.column_stack(
            (np.arange(n_nodes), np.fromiter(map(np.random.choice, adj.tolil().rows), dtype=np.int32)))

        adj_hidden = edges_to_sparse(hiddeen, n_nodes)
        adj_hidden = adj_hidden.maximum(adj_hidden.T)
        adj_keep = adj - adj_hidden
        candidates = np.transpose(sp.triu(adj_keep).nonzero())

        candidates = candidates[np.logical_not(np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]

        return candidates

    def generate_candidates_addition(self, adj, n_candidates):
        """Generates candidate edge flips for addition (non-edge -> edge).

        :param n_candidates: int
            Number of candidates to generate.
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        n_nodes = self.n_nodes

        candidates = np.random.randint(0, n_nodes, [n_candidates * 5, 2])
        candidates = candidates[candidates[:, 0] < candidates[:, 1]]
        candidates = candidates[adj[candidates[:, 0], candidates[:, 1]].A1 == 0]
        candidates = np.array(list(set(map(tuple, candidates))))
        candidates = candidates[:n_candidates]

        return candidates
