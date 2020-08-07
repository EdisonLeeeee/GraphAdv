import numpy as np
import tensorflow as tf
import networkx as nx
from numba import njit

from tensorflow.keras.losses import sparse_categorical_crossentropy
from graphadv.attack.targeted.targeted_attacker import TargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate

from graphgallery.nn.layers import SGConvolution
from graphgallery.nn.models import SGC
from graphgallery import tqdm, astensor


class SGA(TargetedAttacker):
    def __init__(self, adj, x, labels, idx_train=None, idx_val=None, graph=None, radius=2,
                 seed=None, name=None, device='CPU:0', surrogate=None, surrogate_args={}, **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'SGC', idx_train, idx_val, **kwargs)
        elif not isinstance(surrogate, SGC):
            raise RuntimeError("surrogate model should be the instance of `graphgallery.nn.SGC`.")

        self.radius = radius
        self.similar_nodes = [np.where(labels == class_)[0] for class_ in range(self.n_classes)]

        with tf.device(self.device):
            W, b = surrogate.weights
            X = astensor(self.x)
            self.b = b
            self.XW = X @ W
            self.surrogate = surrogate
            self.SGC = SGConvolution(radius)
            self.loss_fn = sparse_categorical_crossentropy
            del X, W

    def reset(self):
        super().reset()
        # for the added self-loop
        self.selfloop_degree = (self.degree + 1.).astype(self.floatx)
        self.structure_flips = {}
        self.sub_edges = None  # 二阶子图的边
        self.sub_nodes = None  # 二阶子图的结点
        self.weights = None       # 子图稀疏邻接矩阵的值
        self.subgraph = None
        self.edge_lower_bound = None  # 可以修改的边的下界
        self.non_edge_lower_bound = None  # 没连接的边的下界
        self.pos_dict = None
        self.wrong_label = None

    def attack(self, target, n_perturbations=None, reduce_nodes=3, direct_attack=True,
               structure_attack=True, feature_attack=False, compute_A_grad=False, disable=False):

        super().attack(target, n_perturbations, direct_attack, structure_attack, feature_attack)

        logit = self.surrogate.predict(target)
        logit = logit / logit.sum()
        self.wrong_label = np.argmax(logit - np.eye(self.n_classes)[self.target_label])
        self.subgraph_preprocessing(reduce_nodes)

        offset = self.edge_lower_bound
        for _ in tqdm(range(self.n_perturbations), desc='Peturbing Graph', disable=disable):
            with tf.device(self.device):
                weights = astensor(self.weights)
                gradients = self.compute_gradient(weights, compute_A_grad=compute_A_grad)
                gradients *= (-2. * weights) + 1.
                gradients = gradients[offset:]
#                 gradients = gradients[offset:] * (-2. * weights[offset:]) + 1.
                sorted_index = tf.argsort(gradients, direction='DESCENDING').numpy()

            for index in sorted_index:
                index_with_offset = index + offset
                u, v = self.indices[index_with_offset]
                has_edge = self.weights[index_with_offset]
                if has_edge and not self.allow_singleton and (self.selfloop_degree[u] <= 2 or self.selfloop_degree[v] <= 2):
                    continue
                if not self.is_modified_edge(u, v):
                    self.structure_flips[(u, v)] = index_with_offset
                    self.update_subgraph(u, v, index_with_offset)
                    break

    def subgraph_preprocessing(self, reduce_nodes):

        target = self.target
        wrong_label = self.wrong_label
        neighbors = self.adj[target].nonzero()[1]
        wrong_label_nodes = self.similar_nodes[wrong_label]
        sub_edges, sub_nodes = self.ego_subgraph()

        if self.direct_attack or reduce_nodes is not None:
            influencer_nodes = [target]
            wrong_label_nodes = np.setdiff1d(wrong_label_nodes, neighbors)
        else:
            influencer_nodes = neighbors

        self.construct_sub_adj(influencer_nodes, wrong_label_nodes, sub_nodes, sub_edges)

        if reduce_nodes is not None:
            if self.direct_attack:
                influencer_nodes = [target]
                wrong_label_nodes = self.top_k_wrong_labels_nodes(k=self.n_perturbations)

            else:
                influencer_nodes = neighbors
                wrong_label_nodes = self.top_k_wrong_labels_nodes(k=reduce_nodes)

            self.construct_sub_adj(influencer_nodes, wrong_label_nodes, sub_nodes, sub_edges)

        self.pos_dict = {(u, v): pos for pos, (u, v) in enumerate(self.indices)}

    @tf.function
    def SGC_conv(self, XW, adj):
        return self.SGC([XW, adj])

    def compute_gradient(self, weights, compute_A_grad=True):

        if compute_A_grad:
            weights = normalize_GCN(self.indices, weights, self.selfloop_degree)

        with tf.GradientTape() as tape:
            tape.watch(weights)
            if compute_A_grad:
                adj = tf.sparse.SparseTensor(self.indices, weights, self.adj.shape)
            else:
                weights_norm = normalize_GCN_tensor(self.indices, weights, self.selfloop_degree)
                adj = tf.sparse.SparseTensor(self.indices, weights_norm, self.adj.shape)

            output = self.SGC_conv(self.XW, adj)
            logit = tf.nn.softmax(output[self.target] + self.b)
            loss = self.loss_fn(self.target_label, logit) - self.loss_fn(self.wrong_label, logit)

        gradients = tape.gradient(loss, weights)

        return gradients

    def ego_subgraph(self):
        '''选取一个二阶子图'''
        # self.subgraph = nx.ego_graph(self.graph, self.target, self.radius)
        # return list(self.subgraph.edges()), list(self.subgraph.nodes())
        return ego_graph(self.adj, self.target, self.radius)

    def construct_sub_adj(self, influencer_nodes, wrong_label_nodes, sub_nodes, sub_edges):
        '''创建子图的稀疏矩阵，并加上自环'''
        length = len(wrong_label_nodes)
        potential_edges = np.vstack([np.stack([np.tile(infl, length), wrong_label_nodes], axis=1)
                                     for infl in influencer_nodes])

        if len(influencer_nodes) > 1:
            '''间接攻击才需要做这些过滤工作，直接攻击的话没必要浪费时间
            还得考虑是否有自环边，所以需要用加了自环的矩阵去过滤'''

            mask = self.adj[tuple(potential_edges.T)].A1 == 0
            potential_edges = potential_edges[mask]

        nodes = np.union1d(sub_nodes, wrong_label_nodes)
        edge_weights = np.ones(len(sub_edges), dtype=self.floatx)
        non_edge_weights = np.zeros(potential_edges.shape[0], dtype=self.floatx)
        self_loop_weights = np.ones(len(nodes), dtype=self.floatx)
        self_loop = np.stack([nodes, nodes], axis=1)

        self.indices = np.vstack([self_loop, potential_edges[:, [1, 0]], sub_edges, potential_edges])
        self.weights = np.hstack([self_loop_weights, non_edge_weights, edge_weights, non_edge_weights])
        self.edge_lower_bound = self_loop_weights.size + non_edge_weights.size
        self.non_edge_lower_bound = self.edge_lower_bound + edge_weights.size
#         self.edge_lower_bound = self_loop_weights.size
#         self.non_edge_lower_bound = self_loop_weights.size

    def top_k_wrong_labels_nodes(self, k):
        '''在原有的结点范围内根据梯度更加缩小范围'''

        offset = self.non_edge_lower_bound
        with tf.device(self.device):
            weights = astensor(self.weights)
            gradients = self.compute_gradient(weights)
            gradients *= weights - 1.
            gradients = gradients[offset:]

#             gradients = gradients[offset:] * (weights[offset:] - 1.)

        index = tf.argsort(gradients)[:k] + offset
        wrong_label_nodes = self.indices[:, 1][index.numpy()]

#         wrong_label_edges = self.indices[index.numpy()]
#         wrong_label_nodes = list(set(wrong_label_edges.ravel().tolist())- set([self.target]))
        return wrong_label_nodes

        # self.construct_sub_adj(influencer_nodes, nodes_with_wrong_label)
#         '''这一步是为了保持梯度正确 而添加了一些边去矫正'''
#         if self.correct_grad:
#             self.correct_subgraph_grad(nodes_with_wrong_label_list)

    def update_subgraph(self, u, v, index):
        '''根据需要修改的边，修改子图的相关数据'''
        weight = 1.0 - self.weights[index]
        degree_delta = 2. * weight - 1.
        inv_index = self.pos_dict[(v, u)]
        self.weights[index] = self.weights[inv_index] = weight
        self.selfloop_degree[u] += degree_delta
        self.selfloop_degree[v] += degree_delta


def normalize_GCN_tensor(indices, weights, degree):
    row, col = indices.T
    inv_degree = tf.pow(degree, -0.5)
    normed_weights = weights * tf.gather(inv_degree, row) * tf.gather(inv_degree, col)
    return normed_weights


def normalize_GCN(indices, weights, degree):
    inv_degree = np.sqrt(degree)
    row, col = indices.T
    normed_weights = weights / (inv_degree[row] * inv_degree[col])
    return astensor(normed_weights)


@njit
def extra_edges(indices, indptr, last_level, seen, radius):
    edges = []
    for u in last_level:
        nbrs = indices[indptr[u]:indptr[u + 1]]
        nbrs = nbrs[seen[nbrs] == radius]
        for v in nbrs:
            edges.append((u, v))
    return edges


def ego_graph(adj, targets, radius=1):
    '''BFS选取一个以 targets 为中心的k阶子图'''
    if np.ndim(targets) == 0:
        nodes = [targets]
    edges = {}
    start = 0
    N = adj.shape[0]
    seen = np.zeros(N)-1
    seen[nodes] = 0
    last_level = []
    for level in range(radius):
        end = len(nodes)
        while start < end:
            head = nodes[start]
            for u in adj[head].nonzero()[1]:
                if (u, head) not in edges:
                    if seen[u] < 0:
                        nodes.append(u)
                        seen[u] = level+1

                        if level == radius-1:
                            last_level.append(u)

                    edges[(u, head)] = level
                    edges[(head, u)] = level
            start += 1

    if len(last_level):
        e = extra_edges(adj.indices, adj.indptr, np.array(last_level), seen, radius)
    else:
        e = []

    return list(edges.keys()) + e, nodes
