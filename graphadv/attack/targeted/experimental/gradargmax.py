import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

from graphadv.attack.targeted.targeted_attacker import TargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate
from graphgallery.nn.models import SemiSupervisedModel, GCN
from graphgallery import tqdm, astensor, normalize_adj


class GradArgmax(TargetedAttacker):
    def __init__(self, adj, x, labels, idx_train=None, idx_val=None,
                 seed=None, name=None, device='CPU:0', surrogate=None, surrogate_args={}, **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'GCN', idx_train, idx_val, **kwargs)

        elif not isinstance(surrogate, SemiSupervisedModel):
            raise RuntimeError("surrogate model should be instance of `graphgallery.nn.SemiSupervisedModel`.")

        with tf.device(self.device):
            self.surrogate = surrogate
            self.loss_fn = sparse_categorical_crossentropy
            self.tf_x = astensor(self.x)

    def reset(self):
        super().reset()
        self.structure_flips = {}
        self.modified_adj = self.adj.copy()
        self.modified_degree = self.degree.copy()

    def attack(self, target, n_perturbations=None, direct_attack=True,
               structure_attack=True, feature_attack=False, disable=False):

        super().attack(target, n_perturbations, direct_attack, structure_attack, feature_attack)

        if direct_attack and self.n_perturbations == self.degree[target]:
            warnings.warn(
                'GradArgmax only work for removing edges, thus it will make the target node become a singleton for direct attack '
                'and `n_perturbations` equals the degree of target.`n_perturbations` is automatically set to `degree-1`.',
                RuntimeWarning)
            self.n_perturbations -= 1

        target_index = astensor([target])
        target_label = astensor(self.target_label)

        surrogate = self.surrogate

        for _ in tqdm(range(self.n_perturbations), desc='Peturbing Graph', disable=disable):
            adj = astensor(normalize_adj(self.modified_adj))
            indices = adj.indices.numpy()
            gradients = self.compute_gradients(adj, target_index, target_label).numpy()
            gradients = np.minimum(gradients, 0.)
            sorted_index = np.argsort(gradients)
            for index in sorted_index:
                u, v = indices[index]
                if not self.allow_singleton and (self.modified_degree[u] <= 1 or
                                                 self.modified_degree[v] <= 1):
                    continue
                if not self.is_modified_edge(u, v):
                    self.structure_flips[(u, v)] = index
                    self.update_graph(u, v)
                    break

    @tf.function
    def compute_gradients(self, adj, target_index, target_label):
        values = adj.values
        with tf.GradientTape() as tape:
            tape.watch(values)
            logit = self.surrogate([self.tf_x, adj, target_index])
            loss = self.loss_fn(target_label, logit, from_logits=True)

        gradients = tape.gradient(loss, values)
        return gradients

    def update_graph(self, u, v):
        self.modified_adj = self.modified_adj.tolil(copy=False)
        self.modified_adj[u, v] = 0.
        self.modified_adj[v, u] = 0.
        self.modified_adj = self.modified_adj.tocsr(copy=False)
        self.modified_adj.eliminate_zeros()

        self.modified_degree[u] -= 1.
        self.modified_degree[v] -= 1.
