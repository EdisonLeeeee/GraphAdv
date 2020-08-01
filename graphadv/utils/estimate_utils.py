import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from numba import njit

def estimate_loss_with_perturbation_gradient(candidates, adj_matrix, window_size, dim, n_neg_samples=1):
    """Computes the estimated loss using the gradient defined with eigenvalue perturbation.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param window_size: int
        Size of the window
    :param dim: int
        Size of the embedding
    :param n_neg_samples: int
        Number of negative samples
    :return:
    """
    adj_matrix = tf.convert_to_tensor(adj_matrix.toarray())
    
    with tf.GradientTape() as tape:
        tape.watch(adj_matrix)
        deg = tf.reduce_sum(adj_matrix, 1)
        volume = tf.reduce_sum(adj_matrix)

        transition_matrix = adj_matrix / deg[:, None]

        sum_of_powers = transition_matrix
        last = transition_matrix
        for i in range(1, window_size):
            last = tf.matmul(last, transition_matrix)
            sum_of_powers += last

        M = sum_of_powers / deg * volume / (n_neg_samples * window_size)
        logM = tf.math.log(tf.maximum(M, 1.0))

        norm_logM = tf.square(tf.norm(logM, ord=2))
        sp_logM = sp.csr_matrix(logM.numpy())
        _, eigenvecs = sp.linalg.eigsh(sp_logM, dim)
        
        eigenvecs = tf.convert_to_tensor(eigenvecs)
        eigen_vals = tf.reduce_sum(eigenvecs * tf.matmul(logM, eigenvecs), 0)
        loss = tf.sqrt(norm_logM - tf.reduce_sum(tf.square(eigen_vals)))
    
    adj_matrix_grad = tape.gradient(loss, adj_matrix).numpy()
    sig_est_grad = adj_matrix_grad[candidates[:, 0], candidates[:, 1]] + adj_matrix_grad[candidates[:, 1], candidates[:, 0]]
    ignore = sig_est_grad < 0
    sig_est_grad[ignore] = - 1

    return sig_est_grad


@njit
def estimate_loss_with_delta_eigenvals(candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size):
    """Computes the estimated loss using the change in the eigenvalues for every candidate edge flip.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips,
    :param flip_indicator: np.ndarray, shape [?]
        Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :param n_nodes: int
        Number of nodes
    :param dim: int
        Embedding dimension
    :param window_size: int
        Size of the window
    :return: np.ndarray, shape [?]
        Estimated loss for each candidate flip
    """

    loss_est = np.zeros(len(candidates))
    for x in range(len(candidates)):
        i, j = candidates[x]
        vals_est = vals_org + flip_indicator[x] * (
                2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2))

        vals_sum_powers = sum_of_powers(vals_est, window_size)

        loss_ij = np.sqrt(np.sum(np.sort(vals_sum_powers ** 2)[:n_nodes - dim]))
        loss_est[x] = loss_ij

    return loss_est

@njit
def sum_of_powers(x, power):
    """For each x_i, computes \sum_{r=1}^{pow) x_i^r (elementwise sum of powers).

    :param x: shape [?]
        Any vector
    :param pow: int
        The largest power to consider
    :return: shape [?]
        Vector where each element is the sum of powers from 1 to pow.
    """
    n = x.shape[0]
    sum_powers = np.zeros((power, n))
    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)

    return sum_powers.sum(0)
