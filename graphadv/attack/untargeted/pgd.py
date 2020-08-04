import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate
from graphgallery.nn.models import DenseGCN
from graphgallery import tqdm, asintarr, normalize_adj_tensor, astensor


class PGD(UntargetedAttacker):

    '''
        PGD cannot ensure that there is not singleton after attack.
        https://github.com/KaidiXu/GCN_ADV_Train

    '''

    def __init__(self, adj, x, labels, idx_train, idx_val, idx_test,
                 use_real_label=False, surrogate=None, surrogate_args={},
                 seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj=adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'DenseGCN', idx_train, idx_val, **surrogate_args)
        else:
            assert isinstance(surrogate, DenseGCN), 'surrogate model should be the instance of `graphgallery.DenseGCN`.'

        idx_train = asintarr(idx_train)
        idx_val = asintarr(idx_val)
        idx_test = asintarr(idx_test)
#         idx_unlabeled = np.hstack([idx_val, idx_test])
        idx_unlabeled = idx_test

        index_all = np.hstack([idx_train, idx_unlabeled])

        # whether to use the ground-truth label as self-training labels
        if use_real_label:
            self_training_labels = labels[idx_unlabeled]
        else:
            self_training_labels = self.estimate_self_training_labels(surrogate, idx_unlabeled)

        with tf.device(self.device):
            self.tf_index = astensor(index_all, dtype=self.intx)
            self.tf_labels = astensor(np.hstack([labels[idx_train], self_training_labels]), dtype=self.intx)
            self.tf_adj = astensor(self.adj.A, dtype=self.floatx)
            self.tf_x = astensor(self.x, dtype=self.floatx)
            self.complementary = tf.ones_like(self.tf_adj) - tf.eye(self.n_nodes) - 2. * self.tf_adj
            self.loss_fn = tf.keras.losses.sparse_categorical_crossentropy
            self.adj_changes = tf.Variable(tf.zeros((self.n_nodes, self.n_nodes), dtype=self.floatx))
            self.surrogate = surrogate

            # used for CW_loss=True
            self.label_matrix = tf.gather(tf.eye(self.n_classes), self.tf_labels)
            self.indices_real = tf.stack([self.tf_index, self.tf_labels], axis=1)

    def reset(self):
        super().reset()

        with tf.device(self.device):
            self.adj_changes.assign(tf.zeros_like(self.adj_changes))

    def estimate_self_training_labels(self, surrogate, idx_unlabeled):
        self_training_labels = surrogate.predict(idx_unlabeled).argmax(1)
        return self_training_labels.astype('int64')

    def attack(self, n_perturbations=0.05, k=20,
               CW_loss=False, epochs=100, structure_attack=True, feature_attack=False, disable=False):

        super().attack(n_perturbations, structure_attack, feature_attack)

        self.CW_loss = CW_loss
        if CW_loss:
            C = 0.1
        else:
            C = 200

        for epoch in tqdm(range(epochs), desc='Peturbation Training', disable=disable):
            with tf.device(self.device):
                gradients = self.compute_gradients()
                lr = C / np.sqrt(epoch+1)
                self.adj_changes.assign_add(lr * gradients)
            self.projection()

        best_s = self.random_sample(k, disable=disable)
        self.structure_flips = np.transpose(np.where(best_s > 0.))

    @tf.function
    def compute_gradients(self):
        with tf.GradientTape() as tape:
            tape.watch(self.adj_changes)
            adj = self.adj_with_perturbation()
            adj_norm = normalize_adj_tensor(adj)
            logit = self.surrogate([self.tf_x, adj_norm, self.tf_index])
            loss = self.compute_loss(logit)

        gradients = tape.gradient(loss, self.adj_changes)
        return gradients

    def compute_loss(self, logit):

        if self.CW_loss:
            best_wrong_class = tf.argmax(logit - self.label_matrix, axis=1, output_type=self.intx)
            indices_attack = tf.stack([self.tf_index, best_wrong_class], axis=1)
            margin = tf.gather_nd(logit, indices_attack) - tf.gather_nd(logit, self.indices_real) - 50
            loss = tf.minimum(margin, 0.)
            loss = tf.reduce_sum(loss)
        else:
            loss = self.loss_fn(self.tf_labels, logit)
            loss = tf.reduce_mean(loss)

        return loss

    def adj_with_perturbation(self):
        adj_triu = tf.linalg.band_part(self.adj_changes, 0, -1) - tf.linalg.band_part(self.adj_changes, 0, 0)
        adj_changes = adj_triu + tf.transpose(adj_triu)
        adj = self.complementary * adj_changes + self.tf_adj
        return adj

    def projection(self):
        clipped_matrix = self.clip(self.adj_changes)
        n_perturbations = tf.reduce_sum(clipped_matrix)

        if n_perturbations > self.n_perturbations:
            left = tf.reduce_min(self.adj_changes - 1.)
            right = tf.reduce_max(self.adj_changes)
            miu = self.bisection(left, right, epsilon=1e-5)
            clipped_matrix = self.clip(self.adj_changes-miu)
        else:
            pass

        self.adj_changes.assign(clipped_matrix)

    def bisection(self, a, b, epsilon):
        def func(x):
            clipped_matrix = self.clip(self.adj_changes-x)
            return tf.reduce_sum(clipped_matrix) - self.n_perturbations

        miu = a
        while (b-a) > epsilon:
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if func(miu)*func(a) < 0:
                b = miu
            else:
                a = miu
        return miu

    def clip(self, matrix):
        clipped_matrix = tf.clip_by_value(matrix, 0., 1.)
        return clipped_matrix

    def random_sample(self, k=20, disable=False):
        best_loss = -np.inf
        best_s = None
        s = np.triu(self.adj_changes.numpy(), 1)
        for _ in tqdm(range(k), desc='Random Sampling', disable=disable):
            #             sampled = np.random.binomial(1, s)
            random_matrix = np.random.uniform(size=(self.n_nodes, self.n_nodes))
            sampled = np.where(s > random_matrix, 1., 0.)
            if sampled.sum() > self.n_perturbations:
                continue

            with tf.device(self.device):
                self.adj_changes.assign(astensor(sampled, dtype=self.floatx))
                adj = self.adj_with_perturbation()
                adj_norm = normalize_adj_tensor(adj)
                logit = self.surrogate([self.tf_x, adj_norm, self.tf_index])
                loss = self.compute_loss(logit)

            if best_loss < loss:
                best_loss = loss
                best_s = sampled

        return best_s


class MinMax(PGD):
    '''MinMax cannot ensure that there is not singleton after attack.'''

    def __init__(self, adj, x, labels, idx_train, idx_val, idx_test, use_real_label=True, normalize_x=True,
                 surrogate=None, surrogate_lr=1e-3, seed=None, device='CPU:0', **kwargs):

        super().__init__(adj, x, labels, idx_train=idx_train,
                         idx_val=idx_val, idx_test=idx_test, use_real_label=use_real_label,
                         normalize_x=normalize_x, surrogate=surrogate,
                         seed=seed, device=device, **kwargs)

        with tf.device(self.device):
            self.stored_weights = self.surrogate.weights.copy()
            self.optimizer = Adam(surrogate_lr)

    def reset(self):
        super().reset()
        weights = self.surrogate.weights
        # restore surrogate weights
        for w1, w2 in zip(weights, self.stored_weights):
            w1.assign(w2)

        # reset optimizer
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

    def attack(self, n_perturbations=0.05, k=20,  CW_loss=False, epochs=100,
               update_per_epoch=20, structure_attack=True, feature_attack=False, disable=False):

        super(PGD, self).attack(n_perturbations, structure_attack, feature_attack)
        self.CW_loss = CW_loss
        if CW_loss:
            C = 0.1
        else:
            C = 200
        trainable_variables = self.surrogate.trainable_variables

        for epoch in tqdm(range(epochs), desc='MinMax Training', disable=disable):
            with tf.device(self.device):
                if (epoch+1) % update_per_epoch == 0:
                    self.update_surrogate(trainable_variables)

                gradients = self.compute_gradients()
                lr = C / np.sqrt(epoch+1)
                self.adj_changes.assign_add(lr * gradients)
            self.projection()

        best_s = self.random_sample(k)
        self.structure_flips = np.transpose(np.where(best_s > 0.))

    @tf.function
    def update_surrogate(self, trainable_variables):
        with tf.GradientTape() as tape:
            adj = self.adj_with_perturbation()
            adj_norm = normalize_adj_tensor(adj)
            logit = self.surrogate([self.tf_x, adj_norm, self.tf_index])
            loss = self.compute_loss(logit)

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
