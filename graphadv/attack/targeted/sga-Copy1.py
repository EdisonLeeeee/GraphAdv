import numpy as np
import tensorflow as tf
import networkx as nx

from graphadv.attack.targeted.targeted_attacker import TargetedAttacker
from graphadv.utils import train_a_surrogate
from graphgallery.nn.layers import SGConvolution
from graphgallery.nn.models import SGC


class SGA(TargetedAttacker):
    def __init__(self, adj, features, labels, idx_train=None, idx_val=None,
                 graph=None, surrogate=None, seed=None,
                 device='CPU:0', radius=2,  **kwargs):

        super().__init__(adj=adj,
                         features=features,
                         labels=labels,
                         seed=seed,
                         device=device,
                         **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'SGC', idx_train, idx_val, kwargs)
        else:
            assert isinstance(surrogate, SGC), 'surrogate model should be the instance of `graphgallery.SGC`.'

        if kwargs:
            raise ValueError(f'Invalid arguments of `{kwargs.keys()}`.')

        if graph is None:
            graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)

        self.graph = graph
        self.radius = radius
        self.similar_nodes = [np.where(labels == class_)[0] for class_ in range(self.n_classes)]

        with tf.device(self.device):
            W, b = surrogate.weights
            X = tf.convert_to_tensor(features)
            self.b = b
            self.XW = X @ W
            self.surrogate = surrogate
            self.SGC = SGConvolution(radius)
            self.loss_fn = tf.keras.losses.sparse_categorical_crossentropy
            del X, W

    def reset(self):
        super().reset()
        # for the added self-loop
        self.selfloop_degree = (self.degree + 1.).astype('float32')
        self.structure_flips = {}
        self.sub_edges = None  # 二阶子图的边
        self.sub_nodes = None  # 二阶子图的结点
        self.values = None       # 子图稀疏邻接矩阵的值
        self.subgraph = None
        self.edge_lower_bound = None  # 可以修改的边的下界
        self.non_edge_lower_bound = None  # 没连接的边的下界
        self.pos_dict = None
        self.wrong_label = None

    def attack(self, target, n_perturbations=None, reduce_nodes=5, direct_attack=True,
               structure_attack=True, feature_attack=False):

        super().attack(target, n_perturbations, direct_attack, structure_attack, feature_attack)

        logit = self.surrogate.predict(target)
        self.wrong_label = np.argmax(logit - np.eye(self.n_classes)[self.target_label])
        self.subgraph_preprocessing(reduce_nodes)

        for _ in range(self.n_perturbations):
            offset = self.edge_lower_bound
            with tf.device(self.device):
                gradients = self.compute_gradient()[offset:]
            sorted_index = np.argsort(-np.abs(gradients))
            for index in sorted_index:
                index_with_offset = index + offset
                u, v = self.indices[index_with_offset]
                has_edge = self.graph.has_edge(u, v)
                if has_edge and not self.allow_singleton and (self.selfloop_degree[u] <= 2 or self.selfloop_degree[v] <= 2):
                    continue
                if all((any((gradients[index] < 0 and not has_edge,
                             gradients[index] > 0 and has_edge and self.labels[u] == self.labels[v])),
                        not self.is_modified_edge(u, v))):
                    self.structure_flips[(u, v)] = index_with_offset
                    self.update_subgraph(u, v, index_with_offset, has_edge)
                    break

    def subgraph_preprocessing(self, reduce_nodes):

        target = self.target
        wrong_label = self.wrong_label
        neighbors = list(self.graph.neighbors(target))
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

    def compute_gradient(self):
        values = tf.convert_to_tensor(self.values)
        with tf.GradientTape() as tape:
            tape.watch(values)
            values_norm = weights_normalize_GCN(self.indices, values, self.selfloop_degree)
            adj = tf.sparse.SparseTensor(self.indices, values_norm, self.adj.shape)
            output = self.SGC_conv(self.XW, adj)
            logit = tf.nn.softmax(output[self.target] + self.b)
            loss = -self.loss_fn(self.target_label, logit) + self.loss_fn(self.wrong_label, logit)
        gradients = tape.gradient(loss, values)
        return gradients.numpy()

    def ego_subgraph(self):
        '''选取一个二阶子图'''
        self.subgraph = nx.ego_graph(self.graph, self.target, self.radius)
        return list(self.subgraph.edges()), list(self.subgraph.nodes())

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
        edge_weights = np.ones(len(sub_edges), dtype=np.float32)
        non_edge_weights = np.zeros(potential_edges.shape[0], dtype=np.float32)
        self_loop_weights = np.ones(len(nodes), dtype=np.float32)
        self_loop = np.stack([nodes, nodes], axis=1)

        self.indices = np.vstack([self_loop, potential_edges[:, [1, 0]], sub_edges, potential_edges])
        self.values = np.hstack([self_loop_weights, non_edge_weights, edge_weights, non_edge_weights])
        self.edge_lower_bound = self_loop_weights.size + non_edge_weights.size
        self.non_edge_lower_bound = self.edge_lower_bound + edge_weights.size

    def top_k_wrong_labels_nodes(self, k):
        '''在原有的结点范围内根据梯度更加缩小范围'''

        offset = self.non_edge_lower_bound
        with tf.device(self.device):
            gradients = self.compute_gradient()[offset:]
        index = np.argsort(gradients)[:k] + offset
        wrong_label_nodes = self.indices[:, 1][index]
        return wrong_label_nodes

        # self.construct_sub_adj(influencer_nodes, nodes_with_wrong_label)
#         '''这一步是为了保持梯度正确 而添加了一些边去矫正'''
#         if self.correct_grad:
#             self.correct_subgraph_grad(nodes_with_wrong_label_list)

    def update_subgraph(self, u, v, index, has_edge):
        '''根据需要修改的边，修改子图的相关数据'''
        weight = 1.0 - has_edge
        degree_delta = 2. * weight - 1.
        inv_index = self.pos_dict[(v, u)]
        self.values[index] = self.values[inv_index] = weight
        self.selfloop_degree[u] += degree_delta
        self.selfloop_degree[v] += degree_delta


def weights_normalize_GCN(indices, weights, degree):
    row, col = indices.T
    inv_degree = tf.pow(degree, -0.5)
    normed_weights = weights * tf.gather(inv_degree, row) * tf.gather(inv_degree, col)
    return normed_weights


# def weights_normalize_GCN(indices, weights, degree):
#     inv_degree = np.sqrt(degree)
#     row, col = indices.T
#     normed_weights = weights / (inv_degree[row] * inv_degree[col])
#     return normed_weights
