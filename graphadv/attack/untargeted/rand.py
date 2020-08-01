import random
import numpy as np
import networkx as nx
from graphgallery import tqdm
from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker


class RAND(UntargetedAttacker):
    def __init__(self, adj, graph=None, seed=None, name=None, **kwargs):
        super().__init__(adj, seed=seed, name=name, **kwargs)

        if graph is None:
            graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)

        self.graph = graph
        self.nodes_set = set(range(self.n_nodes))

    def reset(self):
        super().reset()
        self.structure_flips = {}
        self.modified_degree = self.degree.copy()

    def attack(self, n_perturbations=0.05, structure_attack=True, feature_attack=False, disable=False):

        super().attack(n_perturbations, structure_attack, feature_attack)

        influencer_nodes = list(self.nodes_set)
        structure_flips = self.structure_flips
        random_list = np.random.choice(2, self.n_perturbations) * 2 - 1

        for remove_or_insert in tqdm(random_list, desc='Peturbing Graph', disable=disable):
            if remove_or_insert > 0:
                edge = self.add_an_edge(influencer_nodes)

                while edge is None:
                    edge = self.add_an_edge(influencer_nodes)

            else:
                edge = self.del_an_edge(influencer_nodes)
                while edge is None:
                    edge = self.del_an_edge(influencer_nodes)

            structure_flips[edge] = 1.0
            u, v = edge
            self.modified_degree[u] += remove_or_insert
            self.modified_degree[v] += remove_or_insert

    def add_an_edge(self, influencer_nodes):
        u = random.choice(influencer_nodes)
        # assume that the graph has not self-loops
        potential_nodes = list(self.nodes_set - set(self.graph.neighbors(u)))

        if len(potential_nodes) == 0:
            return None

        v = random.choice(potential_nodes)

        if self.is_modified_edge(u, v):
            return (u, v)
        else:
            return None

    def del_an_edge(self, influencer_nodes):
        u = random.choice(influencer_nodes)
        # assume that the graph has not self-loops
        potential_nodes = list(set(self.graph.neighbors(u)))

        if len(potential_nodes) == 0:
            return None

        v = random.choice(potential_nodes)

        if not self.allow_singleton and (self.modified_degree[u] <= 1 or self.modified_degree[v] <= 1):
            return None

        if self.is_modified_edge(u, v):
            return (u, v)
        else:
            return None
