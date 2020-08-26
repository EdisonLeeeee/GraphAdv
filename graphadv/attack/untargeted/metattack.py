import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform, zeros
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.losses import SparseCategoricalCrossentropy


from graphadv import is_binary
from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate
from graphadv.utils.graph_utils import likelihood_ratio_filter
from graphgallery.nn.models import DenseGCN
from graphgallery import tqdm, asintarr, normalize_adj_tensor, astensor


# cora lr=0.1, citeseer lr=0.01, lambda_=1. reaches best result
class BaseMeta(UntargetedAttacker):
    '''Base model for Mettack.'''

    def __init__(self, adj, x, labels,
                 idx_train, idx_unlabeled,
                 hidden_layers, use_relu, self_training_labels=None,
                 seed=None, name=None, device='CPU:0', **kwargs):

        super().__init__(adj=adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)
        adj, x, labels = self.adj, self.x, self.labels

        idx_train = asintarr(idx_train)
        idx_unlabeled = asintarr(idx_unlabeled)

        if self_training_labels is None:
            surrogate = DenseGCN(adj, x, labels, device='GPU', norm_x=None, seed=None)
            surrogate.build(16, activations='relu' if use_relu else None)
            his = surrogate.train(idx_train, verbose=0, epochs=200, save_best=False)
            self_training_labels = surrogate.predict(idx_unlabeled).argmax(1)

        self.ll_ratio = None
        # mettack can also conduct feature attack
        self.allow_feature_attack = True

        with tf.device(self.device):
            self.idx_train = astensor(idx_train, dtype=self.intx)
            self.idx_unlabeled = astensor(idx_unlabeled, dtype=self.intx)
            self.labels_train = astensor(self.labels[idx_train], dtype=self.floatx)
            self.self_training_labels = astensor(self_training_labels, dtype=self.floatx)
            self.tf_adj = astensor(adj.A, dtype=self.floatx)
            self.tf_x = astensor(x, dtype=self.floatx)
            self.build(hidden_layers=hidden_layers)
            self.use_relu = use_relu
            self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)

            self.adj_changes = tf.Variable(tf.zeros_like(self.tf_adj))
            self.x_changes = tf.Variable(tf.zeros_like(self.tf_x))

    def reset(self):
        super().reset()
        self.structure_flips = []
        self.feature_flips = []

        with tf.device(self.device):
            self.adj_changes.assign(tf.zeros_like(self.tf_adj))
            self.x_changes.assign(tf.zeros_like(self.tf_x))

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

    def log_likelihood_constraint(self, adj, modified_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        """
        t_d_min = tf.constant(2., dtype=self.floatx)
        t_possible_edges = tf.constant(np.array(np.triu(np.ones([self.n_nodes, self.n_nodes]), k=1).nonzero()).T,
                                       dtype=self.intx)
        allowed_mask, current_ratio = likelihood_ratio_filter(t_possible_edges, modified_adj,
                                                              adj, t_d_min, ll_cutoff)

        return allowed_mask, current_ratio

    @tf.function
    def get_perturbed_adj(self, adj, adj_changes):
        adj_changes_square = adj_changes - tf.linalg.band_part(adj_changes, 0, 0)
#         adj_changes_sym = adj_changes_square + tf.transpose(adj_changes_square)
        clipped_adj_changes = self.clip(adj_changes_square)
        return adj + clipped_adj_changes

    @tf.function
    def get_perturbed_x(self, x, x_changes):
        return x + self.clip(x_changes)

    def forward(self, x, adj):
        h = x
        for w in self.weights:
            h = adj @ h @ w
            if self.use_relu:
                h = relu(h)
        return h

    def structure_score(self, modified_adj, adj_grad, ll_constraint=None, ll_cutoff=None):
        adj_meta_grad = adj_grad * (-2. * modified_adj + 1.)
        # Make sure that the minimum entry is 0.
        adj_meta_grad -= tf.reduce_min(adj_meta_grad)

        if not self.allow_singleton:
            # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad *= singleton_mask

        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, self.tf_adj, ll_cutoff)
            adj_meta_grad = adj_meta_grad * allowed_mask

        return tf.reshape(adj_meta_grad, [-1])

    def feature_score(self, modified_x, x_grad):
        x_meta_grad = x_grad * (-2. * modified_x + 1.)
        x_meta_grad -= tf.reduce_min(x_meta_grad)
        return tf.reshape(x_meta_grad, [-1])

    def clip(self, matrix):
        clipped_matrix = tf.clip_by_value(matrix, -1., 1.)
        return clipped_matrix


class Metattack(BaseMeta):

    def __init__(self,  adj, x, labels,
                 idx_train, idx_unlabeled,
                 lr=0.1, epochs=100,
                 momentum=0.9, lambda_=0.,
                 hidden_layers=[16], use_relu=True, self_training_labels=None,
                 seed=None, name=None, device='CPU:0', **kwargs):

        super().__init__(adj, x, labels,
                         idx_train, idx_unlabeled,
                         hidden_layers=hidden_layers, use_relu=use_relu,
                         self_training_labels=self_training_labels,
                         seed=seed, name=name, device=device, **kwargs)

        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.lambda_ = lambda_

        if lambda_ not in (0., 0.5, 1.):
            raise ValueError('Invalid value of `lanbda_`, allowed values [0: (meta-self), 1: (meta-train), 0.5: (meta-both)].')

    def build(self, hidden_layers):

        weights, velocities = [], []
        zeros_initializer = zeros()

        pre_hid = self.n_attrs
        for hid in hidden_layers + [self.n_classes]:
            shape = (pre_hid, hid)
            # use zeros_initializer temporary to save time
            weight = tf.Variable(zeros_initializer(shape=shape, dtype=self.floatx))
            w_velocity = tf.Variable(zeros_initializer(shape=shape, dtype=self.floatx))

            weights.append(weight)
            velocities.append(w_velocity)

            pre_hid = hid

        self.weights, self.velocities = weights, velocities

    def initialize(self):
        w_initializer = glorot_uniform()
        zeros_initializer = zeros()

        for w, wv in zip(self.weights, self.velocities):
            w.assign(w_initializer(w.shape, dtype=self.floatx))
            wv.assign(zeros_initializer(wv.shape, dtype=self.floatx))

    @tf.function
    def train_step(self, x, adj, index, labels):
        with tf.GradientTape() as tape:
            output = self.forward(x, adj)
            logit = tf.gather(output, index)
            loss = self.loss_fn(labels, logit)

        weight_grads = tape.gradient(loss, self.weights)
        return weight_grads

    def inner_train(self, adj, x):

        self.initialize()
        adj_norm = normalize_adj_tensor(adj)

        for _ in range(self.epochs):
            weight_grads = self.train_step(x, adj_norm, self.idx_train, self.labels_train)

            for v, g in zip(self.velocities, weight_grads):
                v.assign(self.momentum * v + g)

            for w, v in zip(self.weights, self.velocities):
                w.assign_sub(self.lr * v)

    @tf.function
    def meta_grad(self):

        modified_adj, modified_x = self.tf_adj, self.tf_x
        persistent = self.structure_attack and self.feature_attack

        with tf.GradientTape(persistent=persistent) as tape:
            if self.structure_attack:
                modified_adj = self.get_perturbed_adj(self.tf_adj, self.adj_changes)

            if self.feature_attack:
                modified_x = self.get_perturbed_x(self.tf_x, self.x_changes)

            adj_norm = normalize_adj_tensor(modified_adj)
            output = self.forward(modified_x, adj_norm) / 5.0
            logit_labeled = tf.gather(output, self.idx_train)
            logit_unlabeled = tf.gather(output, self.idx_unlabeled)

            loss_labeled = self.loss_fn(self.labels_train, logit_labeled)
            loss_unlabeled = self.loss_fn(self.self_training_labels, logit_unlabeled)

            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        adj_grad, x_grad = None, None

        if self.structure_attack:
            adj_grad = tape.gradient(attack_loss, self.adj_changes)

        if self.feature_attack:
            x_grad = tape.gradient(attack_loss, self.x_changes)

        return adj_grad, x_grad

    def attack(self, n_perturbations=0.05, structure_attack=True, feature_attack=False,
               ll_constraint=False, ll_cutoff=0.004, disable=False):

        super().attack(n_perturbations, structure_attack, feature_attack)

        if ll_constraint:
            raise NotImplementedError('`log_likelihood_constraint` has not been well tested.'
                                      ' Please set `ll_constraint=False` to achieve a better performance.')

        if feature_attack and not is_binary(self.x):
            raise ValueError("Attacks on the node features are currently only supported for binary attributes.")

        with tf.device(self.device):
            modified_adj, modified_x = self.tf_adj, self.tf_x
            adj_changes, x_changes = self.adj_changes, self.x_changes
            structure_flips, feature_flips = self.structure_flips, self.feature_flips

            for _ in tqdm(range(self.n_perturbations), desc='Peturbing Graph', disable=disable):

                if structure_attack:
                    modified_adj = self.get_perturbed_adj(self.tf_adj, adj_changes)

                if feature_attack:
                    modified_x = self.get_perturbed_x(self.tf_x, x_changes)

                self.inner_train(modified_adj, modified_x)

                adj_grad, x_grad = self.meta_grad()

                adj_meta_score = tf.constant(0.0)
                x_meta_score = tf.constant(0.0)

                if structure_attack:
                    adj_meta_score = self.structure_score(modified_adj, adj_grad, ll_constraint, ll_cutoff)

                if feature_attack:
                    x_meta_score = self.feature_score(modified_x, x_grad)

                if tf.reduce_max(adj_meta_score) >= tf.reduce_max(x_meta_score):
                    adj_meta_argmax = tf.argmax(adj_meta_score)
                    row, col = divmod(adj_meta_argmax.numpy(), self.n_nodes)
                    adj_changes[row, col].assign(-2. * modified_adj[row, col] + 1.)
                    adj_changes[col, row].assign(-2. * modified_adj[col, row] + 1.)
                    structure_flips.append((row, col))
                else:
                    x_meta_argmax = tf.argmax(x_meta_score)
                    row, col = divmod(x_meta_argmax.numpy(), self.n_attrs)
                    x_changes[row, col].assign(-2 * modified_x[row, col] + 1)
                    feature_flips.append((row, col))


class MetaApprox(BaseMeta):

    def __init__(self,  adj, x, labels,
                 idx_train, idx_unlabeled,
                 lr=0.1, epochs=100, lambda_=0.,
                 hidden_layers=[16], use_relu=True, self_training_labels=None,
                 seed=None, name=None, device='CPU:0', **kwargs):

        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_

        if lambda_ not in (0., 0.5, 1.):
            raise ValueError('Invalid value of `lanbda_`, allowed values [0: (meta-self), 1: (meta-train), 0.5: (meta-both)].')

        super().__init__(adj, x, labels,
                         idx_train, idx_unlabeled,
                         hidden_layers=hidden_layers, use_relu=use_relu,
                         self_training_labels=self_training_labels,
                         seed=seed, name=name, device=device, **kwargs)

    def build(self, hidden_layers):

        weights = []
        zeros_initializer = zeros()

        pre_hid = self.n_attrs
        for hid in hidden_layers + [self.n_classes]:
            shape = (pre_hid, hid)
            # use zeros_initializer temporary to save time
            weight = tf.Variable(zeros_initializer(shape=shape, dtype=self.floatx))
            weights.append(weight)
            pre_hid = hid

        self.weights = weights
        self.adj_grad_sum = tf.Variable(tf.zeros_like(self.tf_adj))
        self.x_grad_sum = tf.Variable(tf.zeros_like(self.tf_x))
        self.optimizer = Adam(self.lr, epsilon=1e-8)

    def initialize(self):

        w_initializer = glorot_uniform()
        zeros_initializer = zeros()

        for w in self.weights:
            w.assign(w_initializer(w.shape, dtype=self.floatx))

        if self.structure_attack:
            self.adj_grad_sum.assign(zeros_initializer(self.adj_grad_sum.shape, dtype=self.floatx))

        if self.feature_attack:
            self.x_grad_sum.assign(zeros_initializer(self.x_grad_sum.shape, dtype=self.floatx))

        # reset optimizer
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

    @tf.function
    def meta_grad(self):
        self.initialize()

        modified_adj, modified_x = self.tf_adj, self.tf_x
        adj_grad_sum, x_grad_sum = self.adj_grad_sum, self.x_grad_sum
        optimizer = self.optimizer

        for _ in tf.range(self.epochs):

            with tf.GradientTape(persistent=True) as tape:
                if self.structure_attack:
                    modified_adj = self.get_perturbed_adj(self.tf_adj, self.adj_changes)

                if self.feature_attack:
                    modified_x = self.get_perturbed_x(self.tf_x, self.x_changes)

                adj_norm = normalize_adj_tensor(modified_adj)
                output = self.forward(modified_x, adj_norm) / 5.0
                logit_labeled = tf.gather(output, self.idx_train)
                logit_unlabeled = tf.gather(output, self.idx_unlabeled)

                loss_labeled = self.loss_fn(self.labels_train, logit_labeled)
                loss_unlabeled = self.loss_fn(self.self_training_labels, logit_unlabeled)

                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            adj_grad, x_grad = None, None

            gradients = tape.gradient(loss_labeled, self.weights)
            optimizer.apply_gradients(zip(gradients, self.weights))

            if self.structure_attack:
                adj_grad = tape.gradient(attack_loss, self.adj_changes)
                adj_grad_sum.assign_add(adj_grad)

            if self.feature_attack:
                x_grad = tape.gradient(attack_loss, self.x_changes)
                x_grad_sum.assign_add(x_grad)

            del tape

        return adj_grad_sum, x_grad_sum

    def attack(self, n_perturbations=0.05, structure_attack=True, feature_attack=False,
               ll_constraint=False, ll_cutoff=0.004, disable=False):

        super().attack(n_perturbations, structure_attack, feature_attack)

        if ll_constraint:
            raise NotImplementedError('`log_likelihood_constraint` has not been well tested.'
                                      ' Please set `ll_constraint=False` to achieve a better performance.')

        if feature_attack and not is_binary(self.x):
            raise ValueError("Attacks on the node features are currently only supported for binary attributes.")

        with tf.device(self.device):
            modified_adj, modified_x = self.tf_adj, self.tf_x
            adj_changes, x_changes = self.adj_changes, self.x_changes
            structure_flips, feature_flips = self.structure_flips, self.feature_flips

            for _ in tqdm(range(self.n_perturbations), desc='Peturbing Graph', disable=disable):

                adj_grad, x_grad = self.meta_grad()

                adj_meta_score = tf.constant(0.0)
                x_meta_score = tf.constant(0.0)

                if structure_attack:
                    modified_adj = self.get_perturbed_adj(self.tf_adj, adj_changes)
                    adj_meta_score = self.structure_score(modified_adj, adj_grad, ll_constraint, ll_cutoff)

                if feature_attack:
                    modified_x = self.get_perturbed_x(self.tf_x, x_changes)
                    x_meta_score = self.feature_score(modified_x, feature_grad)

                if tf.reduce_max(adj_meta_score) >= tf.reduce_max(x_meta_score):
                    adj_meta_argmax = tf.argmax(adj_meta_score)
                    row, col = divmod(adj_meta_argmax.numpy(), self.n_nodes)
                    adj_changes[row, col].assign(-2. * modified_adj[row, col] + 1.)
                    adj_changes[col, row].assign(-2. * modified_adj[col, row] + 1.)
                    structure_flips.append((row, col))
                else:
                    x_meta_argmax = tf.argmax(x_meta_score)
                    row, col = divmod(x_meta_argmax.numpy(), self.n_attrs)
                    x_changes[row, col].assign(-2 * modified_x[row, col] + 1)
                    feature_flips.append((row, col))
