import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import sparse_categorical_crossentropy
from graphadv.attack.targeted.targeted_attacker import TargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate

from graphgallery.nn.layers import SGConvolution
from graphgallery.nn.models import SGC
from graphgallery import tqdm, astensor, ego_graph


class SGA(TargetedAttacker):
    def __init__(self, adj, x, labels, idx_train=None, hops=2,
                 seed=None, name=None, device='CPU:0', surrogate=None, surrogate_args={}, **kwargs):
        
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'SGC', idx_train, **surrogate_args)
        elif not isinstance(surrogate, SGC):
            raise RuntimeError("surrogate model should be the instance of `graphgallery.nn.SGC`.")
            
        self.hops = hops
        # nodes with the same class labels
        self.similar_nodes = [np.where(labels == c)[0] for c in range(self.n_classes)]

        with tf.device(self.device):
            W, b = surrogate.weights
            X = astensor(x)
            self.b = b
            self.XW = X @ W
            self.surrogate = surrogate
            self.SGC = SGConvolution(hops)
            self.loss_fn = sparse_categorical_crossentropy

    def reset(self):
        super().reset()
        # for the added self-loop
        self.selfloop_degree = (self.degree + 1.).astype(self.floatx)
        self.structure_flips = {}
        self.wrong_label = None

    def attack(self, target, n_perturbations=None, logit=None, reduced_nodes=3, direct_attack=True,
               structure_attack=True, feature_attack=False, disable=False):

        super().attack(target, n_perturbations, direct_attack, structure_attack, feature_attack)

        if logit is None:
            logit = self.surrogate.predict(target).ravel()
            
        top2 = logit.argsort()[-2:]
        self.wrong_label = np.setdiff1d(logit.argsort()[-2:], self.target_label)[0]
        assert self.wrong_label != self.target_label
        
        self.subgraph_preprocessing(reduced_nodes)
        offset = self.edge_weights.shape[0]
        
        with tf.device(self.device):
            for _ in tqdm(range(self.n_perturbations), desc='Peturbing Graph', disable=disable):
                edge_grad, non_edge_grad = self.compute_gradient(norm=False)
                edge_grad *= (-2 * self.edge_weights + 1)
                non_edge_grad *= (-2 * self.non_edge_weights + 1)
                gradients = tf.concat([edge_grad, non_edge_grad], axis=0) 
                sorted_indices = tf.argsort(gradients, direction="DESCENDING")

                for index in sorted_indices:
                    if index<offset:
                        u, v = self.edge_index[index]
                        add = False
                        if not self.allow_singleton and (self.selfloop_degree[u] <= 2 or self.selfloop_degree[v] <= 2):
                            continue                        
                    else:
                        index -= offset
                        u, v = self.non_edge_index[index]
                        add = True

                    if not self.is_modified_edge(u, v):
                        self.structure_flips[(u, v)] = _
                        self.update_subgraph(u, v, index, add=add)     
                        break
                        
    def subgraph_preprocessing(self, reduced_nodes=None):
        target = self.target
        wrong_label = self.wrong_label
        neighbors = self.adj[target].nonzero()[1]
        wrong_label_nodes = self.similar_nodes[wrong_label]
        sub_edges, sub_nodes = self.ego_subgraph()
        
        if self.direct_attack or reduced_nodes is not None:
            influence_nodes = [target]
            wrong_label_nodes = np.setdiff1d(wrong_label_nodes, neighbors)
        else:
            influence_nodes = neighbors

        self.construct_sub_adj(influence_nodes, wrong_label_nodes, sub_nodes, sub_edges)

        if reduced_nodes is not None:
            if self.direct_attack:
                influence_nodes = [target]
                wrong_label_nodes = self.top_k_wrong_labels_nodes(k=self.n_perturbations)

            else:
                influence_nodes = neighbors
                wrong_label_nodes = self.top_k_wrong_labels_nodes(k=reduced_nodes)

            self.construct_sub_adj(influence_nodes, wrong_label_nodes, sub_nodes, sub_edges)

    @tf.function
    def SGC_conv(self, XW, adj):
        return self.SGC([XW, adj])

    def compute_gradient(self, eps=2.24, norm=False):

        edge_weights = self.edge_weights
        non_edge_weights = self.non_edge_weights
        self_loop_weights = self.self_loop_weights
        
        if norm:
            edge_weights = normalize_GCN(self.edge_index, edge_weights, self.selfloop_degree)
            non_edge_weights = normalize_GCN(self.non_edge_index, non_edge_weights, self.selfloop_degree)
            self_loop_weights = normalize_GCN(self.self_loop, self_loop_weights, self.selfloop_degree)
            
        with tf.GradientTape() as tape:
            tape.watch([edge_weights, non_edge_weights])

            weights = tf.concat([edge_weights, 
                                 edge_weights,
                                 non_edge_weights,                                  
                                 non_edge_weights, 
                                 self_loop_weights], axis=0)

            if norm:
                adj = tf.sparse.SparseTensor(self.indices, weights, self.adj.shape)
            else:
                weights_norm = normalize_GCN(self.indices, weights, self.selfloop_degree)
                adj = tf.sparse.SparseTensor(self.indices, weights_norm, self.adj.shape)
                
            output = self.SGC_conv(self.XW, adj)
            logit = output[self.target] + self.b
            logit = tf.nn.softmax(logit/eps)
#             loss = self.loss_fn(self.target_label, logit) - self.loss_fn(self.wrong_label, logit)
            loss = self.loss_fn(self.target_label, logit) - self.loss_fn(self.wrong_label, logit)

            
        gradients = tape.gradient(loss, [edge_weights, non_edge_weights])
        return gradients

    def ego_subgraph(self):
        return ego_graph(self.adj, self.target, self.hops)

    def construct_sub_adj(self, influence_nodes, wrong_label_nodes, sub_nodes, sub_edges):
        length = len(wrong_label_nodes)
        potential_edges = np.vstack([np.stack([np.tile(infl, length), wrong_label_nodes], axis=1)
                                     for infl in influence_nodes])

        if len(influence_nodes) > 1:
            # TODO: considering self-loops
            mask = self.adj[tuple(potential_edges.T)].A1 == 0
            potential_edges = potential_edges[mask]

        nodes = np.union1d(sub_nodes, wrong_label_nodes)
        edge_weights = np.ones(sub_edges.shape[0], dtype=self.floatx)
        non_edge_weights = np.zeros(potential_edges.shape[0], dtype=self.floatx)
        self_loop_weights = np.ones(nodes.shape[0], dtype=self.floatx)
        self_loop = np.stack([nodes, nodes], axis=1)

        self.indices = np.vstack([sub_edges, sub_edges[:, [1,0]], potential_edges, potential_edges[:, [1,0]], self_loop])
        self.edge_weights = tf.Variable(edge_weights, dtype=self.floatx)
        self.non_edge_weights = tf.Variable(non_edge_weights, dtype=self.floatx)
        self.self_loop_weights = astensor(self_loop_weights, dtype=self.floatx)
        self.edge_index = sub_edges
        self.non_edge_index = potential_edges  
        self.self_loop = self_loop

    def top_k_wrong_labels_nodes(self, k):
        with tf.device(self.device):
            _, non_edge_grad = self.compute_gradient(norm=True)
            _, index = tf.math.top_k(non_edge_grad, k=k, sorted=False)

        wrong_label_nodes = self.non_edge_index[:, 1][index.numpy()]
        return wrong_label_nodes

    def update_subgraph(self, u, v, index, add=True):
        if add:
            self.non_edge_weights[index].assign(1.0)
            degree_delta = 1.0
        else:
            self.edge_weights[index].assign(0.0)
            degree_delta = -1.0
            
        self.selfloop_degree[u] += degree_delta
        self.selfloop_degree[v] += degree_delta

def normalize_GCN(indices, weights, degree):
    row, col = indices.T
    inv_degree = tf.pow(degree, -0.5)
    normed_weights = weights * tf.gather(inv_degree, row) * tf.gather(inv_degree, col)
    return normed_weights