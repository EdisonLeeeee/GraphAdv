import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

from graphadv import is_binary
from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate
from graphgallery.nn.models import DenseGCN
from graphgallery import tqdm, astensor, normalize_adj_tensor


class FGSM(UntargetedAttacker):
    def __init__(self, adj, x, labels, idx_train=None, surrogate=None, surrogate_args={},
                 seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'DenseGCN', idx_train, **surrogate_args)
        elif not isinstance(surrogate, DenseGCN):
            raise RuntimeError("surrogate model should be the instance of `graphgallery.nn.DenseGCN`.")

        self.allow_feature_attack = True

        with tf.device(self.device):
            self.surrogate = surrogate
            self.tf_idx_train = astensor(idx_train)
            self.tf_labels = astensor(labels[idx_train])
            self.loss_fn = sparse_categorical_crossentropy

    def reset(self):
        super().reset()
        self.structure_flips = []
        self.attribute_flips = []

        with tf.device(self.device):
            self.modified_adj = tf.Variable(self.adj.A, dtype=self.floatx)
            self.modified_x = tf.Variable(x, dtype=self.floatx)

    def attack(self, n_perturbations=0.05, symmetric=True,
               structure_attack=True, feature_attack=False, disable=False):

        super().attack(n_perturbations, structure_attack, feature_attack)

        if feature_attack:
            assert is_binary(self.x)

        modified_adj, modified_x = self.modified_adj, self.modified_x

        for _ in tqdm(range(self.n_perturbations), desc='Peturbing Graph', disable=disable):

            with tf.device(self.device):
                adj_grad, x_grad = self.compute_gradients(modified_adj, modified_x, self.tf_idx_train)

                adj_grad_score = tf.constant(0.0)
                x_grad_score = tf.constant(0.0)

                if structure_attack:

                    if symmetric:
                        adj_grad += tf.transpose(adj_grad)

                    adj_grad_score = self.structure_score(modified_adj, adj_grad)

                if feature_attack:
                    x_grad_score = self.feature_score(modified_x, x_grad)

                if tf.reduce_max(adj_grad_score) >= tf.reduce_max(x_grad_score):
                    adj_grad_argmax = tf.argmax(adj_grad_score)
                    row, col = divmod(adj_grad_argmax.numpy(), self.n_nodes)
                    modified_adj[row, col].assign(1. - modified_adj[row, col])
                    modified_adj[col, row].assign(1. - modified_adj[col, row])
                    self.structure_flips.append((row, col))
                else:
                    x_grad_argmax = tf.argmax(x_grad_score)
                    row, col = divmod(x_grad_argmax.numpy(), self.n_features)
                    modified_x[row, col].assign(1. - modified_x[row, col])
                    self.attribute_flips.append((row, col))

    @tf.function
    def structure_score(self, modified_adj, adj_grad):
        adj_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_grad = adj_grad - tf.reduce_min(adj_grad)
        # Filter self-loops
        adj_grad = adj_grad - tf.linalg.band_part(adj_grad, 0, 0)

        if not self.allow_singleton:
            # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_grad = adj_grad * singleton_mask

        return tf.reshape(adj_grad, [-1])

    @tf.function
    def feature_score(self, modified_x, x_grad):

        x_grad = x_grad * (-2. * modified_x + 1.)
        min_grad = tf.reduce_min(x_grad)
        x_grad = x_grad - min_grad

        return tf.reshape(x_grad, [-1])

    @tf.function
    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        Returns
        -------
        tf.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.
        """
        N = self.n_nodes
        degrees = tf.reduce_sum(modified_adj, axis=1)
        degree_one = tf.equal(degrees, 1)
        resh = tf.reshape(tf.tile(degree_one, [N]), [N, N])
        l_and = tf.logical_and(resh, tf.equal(modified_adj, 1))
        logical_and_symmetric = tf.logical_or(l_and, tf.transpose(l_and))
        flat_mask = 1. - tf.cast(logical_and_symmetric, self.floatx)
        return flat_mask

    @tf.function
    def compute_gradients(self, modified_adj, modified_x, index):
        # TODO persistent=False
        with tf.GradientTape(persistent=True) as tape:
            adj_norm = normalize_adj_tensor(modified_adj)
            logit = self.surrogate([modified_x, adj_norm, index])
            loss = self.loss_fn(self.tf_labels, logit, from_logits=True)

        adj_grad, x_grad = None, None
        if self.structure_attack:
            adj_grad = tape.gradient(loss, modified_adj)

        if self.feature_attack:
            x_grad = tape.gradient(loss, modified_x)

        return adj_grad, x_grad
