import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

from graphadv.attack.targeted.targeted_attacker import TargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate
from graphgallery.nn.models import DenseGCN
from graphgallery import tqdm, astensor, normalize_adj_tensor


class FGSM(TargetedAttacker):
    def __init__(self, adj, x, labels, idx_train=None, idx_val=None,
                 seed=None, name=None, device='CPU:0', surrogate=None, surrogate_args={}, **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'DenseGCN', idx_train, idx_val, **surrogate_args)
        elif not isinstance(surrogate, DenseGCN):
            raise RuntimeError("surrogate model should be the instance of `graphgallery.nn.DenseGCN`.")

        with tf.device(self.device):
            self.surrogate = surrogate
            self.loss_fn = sparse_categorical_crossentropy
            self.tf_x = astensor(self.x)

    def reset(self):
        super().reset()
        self.modified_degree = self.degree.copy()
        self.structure_flips = {}

        with tf.device(self.device):
            self.adj_changes = tf.Variable(tf.zeros(self.n_nodes), dtype=self.floatx)
            self.modified_adj = tf.Variable(self.adj.A, dtype=self.floatx)

    def attack(self, target, n_perturbations=None, direct_attack=True,
               structure_attack=True, feature_attack=False, disable=False):

        super().attack(target, n_perturbations, direct_attack, structure_attack, feature_attack)

        if not direct_attack:
            raise NotImplementedError(f'{self.name} does not support indirect attack now.')

        target_index = astensor([target])
        target_label = astensor(self.target_label)

        for _ in tqdm(range(self.n_perturbations), desc='Peturbing Graph', disable=disable):
            with tf.device(self.device):
                gradients = self.compute_gradients(self.modified_adj, self.adj_changes, target_index, target_label)

            modified_row = tf.gather(self.modified_adj, target_index)
            gradients = (gradients * (-2*modified_row + 1)).numpy().ravel()

            sorted_index = np.argsort(-gradients)
            for index in sorted_index:
                u = target
                v = index % self.n_nodes
                has_edge = self.adj[u, v]
                if has_edge and not self.allow_singleton and (self.modified_degree[u] <= 1 or
                                                              self.modified_degree[v] <= 1):
                    continue
                if not self.is_modified_edge(u, v):
                    self.structure_flips[(u, v)] = index
                    self.update_graph(u, v, has_edge)
                    break

    @tf.function
    def compute_gradients(self, modified_adj, adj_changes, target_index, target_label):

        with tf.GradientTape() as tape:
            tape.watch(adj_changes)
            adj = modified_adj + adj_changes
            adj_norm = normalize_adj_tensor(adj)
            logit = self.surrogate([self.tf_x, adj_norm, target_index])
            loss = self.loss_fn(target_label, logit, from_logits=True)

        gradients = tape.gradient(loss, adj_changes)
        return gradients

    def update_graph(self, u, v, has_edge):

        weight = 1. - has_edge
        delta_d = 2. * weight - 1.

        self.modified_adj[u, v].assign(weight)
        self.modified_adj[v, u].assign(weight)

        self.modified_degree[u] += delta_d
        self.modified_degree[v] += delta_d
