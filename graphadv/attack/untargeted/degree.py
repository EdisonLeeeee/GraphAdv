import numpy as np
import scipy.sparse as sp

from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphadv.utils import edges_to_sparse
from graphgallery import tqdm


class Deg(UntargetedAttacker):
    '''
        For each perturbation, inserting or removing an edge based on degree centrality, 
        which is equivalent to the sum of degrees in original graph

    '''

    def __init__(self, adj, name=None, seed=None, **kwargs):
        super().__init__(adj=adj, name=name, seed=seed, **kwargs)
        self.nodes_set = set(range(self.n_nodes))

    def attack(self, n_perturbations=0.05, complement=False,
               addition=True, removel=False,
               structure_attack=True, feature_attack=False):

        super().attack(n_perturbations, structure_attack, feature_attack)
        n_perturbations = self.n_perturbations

        candidates = []
        if addition:
            n_candidates = min(2*n_perturbations, self.n_edges)
            candidates.append(self.generate_candidates_addition(n_candidates))

        if removel:
            candidates.append(self.generate_candidates_removal())

        candidates = np.vstack(candidates)

        deg_argsort = (self.degree[candidates[:, 0]] + self.degree[candidates[:, 1]]).argsort()
        self.structure_flips = candidates[deg_argsort[-n_perturbations:]]

    def generate_candidates_removal(self):
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

        adj_hidden = edges_to_sparse(hiddeen, adj.shape[0])
        adj_hidden = adj_hidden.maximum(adj_hidden.T)
        adj_keep = adj - adj_hidden
        candidates = np.transpose(sp.triu(adj_keep).nonzero())

        candidates = candidates[np.logical_not(np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]

        return candidates

    def generate_candidates_addition(self, n_candidates):
        """Generates candidate edge flips for addition (non-edge -> edge).

        :param n_candidates: int
            Number of candidates to generate.
        :return: np.ndarray, shape [?, 2]
            Candidate set of edge flips
        """
        n_nodes = self.n_nodes
        adj = self.adj

        candidates = np.random.randint(0, n_nodes, [n_candidates * 5, 2])
        candidates = candidates[candidates[:, 0] < candidates[:, 1]]
        candidates = candidates[adj[candidates[:, 0], candidates[:, 1]].A1 == 0]
        candidates = np.array(list(set(map(tuple, candidates))))
        candidates = candidates[:n_candidates]

        return candidates
