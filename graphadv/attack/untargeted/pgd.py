import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax

from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphadv.utils.surrogate_utils import train_a_surrogate
from graphgallery.nn.models import DenseGCN
from graphgallery import tqdm, asintarr, normalize_adj_tensor, astensor


class PGD(UntargetedAttacker):

    '''
        PGD cannot ensure that there is not singleton after attack.
        https://github.com/KaidiXu/GCN_ADV_Train

    '''

    def __init__(self, adj, x, labels, idx_train, idx_unlabeled=None,
                 surrogate=None, surrogate_args={},
                 seed=None, name=None, device='CPU:0', **kwargs):
        
        super().__init__(adj=adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)
        
        adj, x = self.adj, self.x

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'DenseGCN', idx_train, **surrogate_args)
        elif not isinstance(surrogate, DenseGCN):
            raise RuntimeError("surrogate model should be the instance of `graphgallery.nn.DenseGCN`.")

        # poisoning attack in DeepRobust
        if idx_unlabeled is None:
            idx_attack = idx_train
            labels_attack = labels[idx_train]
        else: # Evasion attack in original paper
            idx_unlabeled = asintarr(idx_unlabeled)
            self_training_labels = self.estimate_self_training_labels(surrogate, idx_unlabeled)
            idx_attack = np.hstack([idx_train, idx_unlabeled])
            labels_attack = np.hstack([labels[idx_train], self_training_labels])
            
        with tf.device(self.device):
            self.idx_attack = astensor(idx_attack)
            self.labels_attack = astensor(labels_attack)
            self.tf_adj = astensor(self.adj.A)
            self.tf_x = astensor(x)
            self.complementary = tf.ones_like(self.tf_adj) - tf.eye(self.n_nodes) - 2. * self.tf_adj
            self.loss_fn = SparseCategoricalCrossentropy()
            self.adj_changes = tf.Variable(tf.zeros(adj.shape, dtype=self.floatx))
            self.surrogate = surrogate

            # used for `CW_loss=True`
            self.label_matrix = tf.gather(tf.eye(self.n_classes), self.labels_attack)
            self.range_idx = tf.range(idx_attack.size, dtype=self.intx)
            self.indices_real = tf.stack([self.range_idx, self.labels_attack], axis=1)

    def reset(self):
        super().reset()

        with tf.device(self.device):
            self.adj_changes.assign(tf.zeros_like(self.adj_changes))

    def estimate_self_training_labels(self, surrogate, idx_attack):
        self_training_labels = surrogate.predict(idx_attack).argmax(1)
        return self_training_labels.astype(self.intx)

    def attack(self, n_perturbations=0.05, sample_epochs=20, CW_loss=True, epochs=100,
               structure_attack=True, feature_attack=False, disable=False):

        super().attack(n_perturbations, structure_attack, feature_attack)

        self.CW_loss = CW_loss

        if CW_loss:
            C = 0.1
        else:
            C = 200

        with tf.device(self.device):
            for epoch in tqdm(range(epochs), desc='Peturbation Training', disable=disable):
                gradients = self.compute_gradients(self.idx_attack)
                lr = C / np.sqrt(epoch+1)
                self.adj_changes.assign_add(lr * gradients)
                self.projection()

            best_s = self.random_sample(sample_epochs, disable=disable)
            self.structure_flips = np.transpose(np.where(best_s > 0.))

    @tf.function
    def compute_gradients(self, idx):
        with tf.GradientTape() as tape:
            tape.watch(self.adj_changes)
            adj = self.get_perturbed_adj()
            adj_norm = normalize_adj_tensor(adj)
            logit = self.surrogate([self.tf_x, adj_norm, idx])
            logit = softmax(logit)
            loss = self.compute_loss(logit)

        gradients = tape.gradient(loss, self.adj_changes)
        return gradients

    @tf.function
    def compute_loss(self, logit):

        if self.CW_loss:
            best_wrong_class = tf.argmax(logit - self.label_matrix, axis=1, output_type=self.intx)
            indices_attack = tf.stack([self.range_idx, best_wrong_class], axis=1)
            margin = tf.gather_nd(logit, indices_attack) - tf.gather_nd(logit, self.indices_real) - 0.2
            loss = tf.minimum(margin, 0.)
            loss = tf.reduce_sum(loss)
        else:
            loss = self.loss_fn(self.labels_attack, logit)
            logit = tf.argmax(logit, axis=1, output_type=self.intx)

        return loss

    @tf.function
    def get_perturbed_adj(self):
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
            if func(miu) == 0:
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

    def random_sample(self, sample_epochs=20, disable=False):
        best_loss = -np.inf
        best_s = None
        s = tf.linalg.band_part(self.adj_changes, 0, -1) - tf.linalg.band_part(self.adj_changes, 0, 0)
        for _ in tqdm(range(sample_epochs), desc='Random Sampling', disable=disable):
            random_matrix = tf.random.uniform(shape=(self.n_nodes, self.n_nodes), minval=0., maxval=1.)
            sampled = tf.where(s > random_matrix, 1., 0.)
            if tf.reduce_sum(sampled) > self.n_perturbations:
                continue

            with tf.device(self.device):
                self.adj_changes.assign(sampled)
                adj = self.get_perturbed_adj()
                adj_norm = normalize_adj_tensor(adj)
                logit = self.surrogate([self.tf_x, adj_norm, self.idx_attack])
                logit = softmax(logit)
                loss = self.compute_loss(logit)

            if best_loss < loss:
                best_loss = loss
                best_s = sampled

        return best_s.numpy()

    
class MinMax(PGD):
    '''MinMax cannot ensure that there is not singleton after attack.'''

    def __init__(self, adj, x, labels, idx_train, idx_unlabeled=None,
                 surrogate=None, surrogate_args={}, surrogate_lr=5e-3,
                 seed=None, name=None, device='CPU:0', **kwargs):

        super().__init__(adj, x, labels, idx_train=idx_train, idx_unlabeled=idx_unlabeled, 
                         surrogate=surrogate, surrogate_args=surrogate_args,
                         seed=seed, device=device, **kwargs)

        with tf.device(self.device):
            self.stored_weights = tf.identity_n(self.surrogate.weights)
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

    def attack(self, n_perturbations=0.05, sample_epochs=20,  CW_loss=True, epochs=100,
               update_per_epoch=20, structure_attack=True, feature_attack=False, disable=False):

        super(PGD, self).attack(n_perturbations, structure_attack, feature_attack)

        self.CW_loss = CW_loss

        if CW_loss:
            C = 0.1
        else:
            C = 200

        with tf.device(self.device):

            trainable_variables = self.surrogate.trainable_variables

            for epoch in tqdm(range(epochs), desc='MinMax Training', disable=disable):
                if (epoch+1) % update_per_epoch == 0:
                    self.update_surrogate(trainable_variables, self.idx_attack)
                gradients = self.compute_gradients(self.idx_attack)
                lr = C / np.sqrt(epoch+1)
                self.adj_changes.assign_add(lr * gradients)
                self.projection()

            best_s = self.random_sample(sample_epochs)
            self.structure_flips = np.transpose(np.where(best_s > 0.))

    @tf.function
    def update_surrogate(self, trainable_variables, idx):
        with tf.GradientTape() as tape:
            adj = self.get_perturbed_adj()
            adj_norm = normalize_adj_tensor(adj)
            logit = self.surrogate([self.tf_x, adj_norm, idx])
            logit = softmax(logit)
            loss = self.compute_loss(logit)

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    