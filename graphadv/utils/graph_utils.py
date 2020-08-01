import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.

    Parameters
    ----------
    edges: np.array, shape [P, 2], dtype int, where P is the number of input edges.
        The potential edges.

    adj: sp.sparse_matrix, shape [n_nodes, n_nodes]
        The input adjacency matrix.

    Returns
    -------
    np.array, shape [P, 2], dtype bool:
        A binary vector of length len(edges), False values indicate that the edge at
        the index  generates singleton edges, and should thus be avoided.

    """

    edges = np.asarray(edges)
    if edges.size == 0:
        return edges
    degs = adj.sum(1).A1
    existing_edges = adj.tocsr()[tuple(edges.T)].A1
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:, None]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1

    mask = np.logical_and(edge_degrees[:, 0] != 0, edge_degrees[:, 1] != 0)
    remained_edges = edges[mask]
    return remained_edges

#     zeros = edge_degrees == 0
#     zeros_sum = zeros.sum(1)
#     return zeros_sum == 0


def edges_to_sparse(edges, n_nodes, weights=None):
    """Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    :param edges: array-like, shape [num_edges, 2]
        Array with each row storing indices of an edge as (u, v).
    :param n_nodes: int
        Number of nodes in the resulting graph.
    :param weights: array_like, shape [num_edges], optional, default None
        Weights of the edges. If None, all edges weights are set to 1.
    :return: sp.csr_matrix
        Adjacency matrix in CSR format.
    """
    if weights is None:
        weights = np.ones(edges.shape[0])

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes)).tocsr()


def ravel_multiple_indices(ixs, shape):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    ----------
    ixs: array of ints shape (n, 2)
        The array of n indices that will be flattened.
    shape: list or tuple of ints of length 2
        The shape of the corresponding matrix.
    Returns
    -------
    array of n ints between 0 and shape[0]*shape[1]-1
        The indices on the flattened matrix corresponding to the 2D input indices.
    """
    return ixs[:, 0] * shape[1] + ixs[:, 1]


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    """
    Computes thelog likelihood of the observed Powerlaw distribution given the Powerlaw exponent alpha.
    Parameters
    ----------
    n: int
        The number of samples in the observed distribution whose value is >= d_min.
    alpha: float
        The Powerlaw exponent for which the log likelihood is to be computed.
    sum_log_degrees: float
        The sum of the logs of samples in the observed distribution whose values are >= d_min.
    d_min: int
        The minimum degree to be considered in the Powerlaw computation.
    Returns
    -------
    float
        The log likelihood of the given observed Powerlaw distribution and exponend alpha.
    """
    return n * tf.math.log(alpha) + n * alpha * tf.math.log(d_min) + (alpha + 1) * sum_log_degrees


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    """
    Compute the sum of the logs of samples in the observed distribution whose values are >= d_min for a single edge
    changing in the graph. That is, given that two degrees in the graph change from d_old to d_new respectively
    (resulting from adding or removing a single edge), compute the updated sum of log degrees >= d_min.
    Parameters
    ----------
    sum_log_degrees_before: tf.Tensor of floats of length n
        The sum of log degrees >= d_min before the change.
    n_old: tf.Tensor of ints of length n
        The number of degrees >= d_min before the change.
    d_old: tf.Tensor of ints, shape [n, 2]
        The old (i.e. before change) degrees of the two nodes affected by an edge to be inserted/removed. n corresponds
        to the number of edges for which this will be computed in a vectorized fashion.
    d_new: tf.Tensor of ints, shape [n,2]
        The new (i.e. after the change) degrees of the two nodes affected by an edge to be inserted/removed.
        n corresponds to the number of edges for which this will be computed in a vectorized fashion.
    d_min: int
        The minimum degree considered in the Powerlaw distribution.
    Returns
    -------
    sum_log_degrees_after: tf.Tensor of floats shape (n,)
        The updated sum of log degrees whose values are >= d_min after a potential edge being added/removed.
    new_n: tf.Tensor dtype int shape (n,)
        The updated number of degrees which are >= d_min after a potential edge being added/removed.
    """

    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    # Mask out the degrees whose values are below d_min by multiplying them by 0.
    d_old_in_range = tf.multiply(d_old, tf.cast(old_in_range, self.floatx))
    d_new_in_range = tf.multiply(d_new, tf.cast(new_in_range, self.floatx))

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - tf.reduce_sum(tf.math.log(tf.maximum(d_old_in_range, 1)),
                                                                   axis=1) + tf.reduce_sum(
        tf.math.log(tf.maximum(d_new_in_range, 1)), axis=1)

    # Update the number of degrees >= d_min
    new_n = tf.cast(n_old, self.intx) - tf.math.count_nonzero(old_in_range, axis=1) + tf.math.count_nonzero(new_in_range, axis=1)

    return sum_log_degrees_after, new_n


def compute_alpha(n, sum_log_degrees, d_min):
    """
    Compute the maximum likelihood value of the Powerlaw exponent alpha of the degree distribution.
    Parameters
    ----------
    n: int
        The number of degrees >= d_min
    sum_log_degrees: float
        The sum of log degrees >= d_min
    d_min: int
        The minimum degree considered in the Powerlaw distribution.
    Returns
    -------
    alpha: float
        The maximum likelihood estimate of the Powerlaw exponent alpha.
    """
    return n / (sum_log_degrees - n * tf.math.log(d_min - 0.5)) + 1


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    Parameters
    ----------
    degree_sequence: tf.Tensor dtype int shape (N,)
        Observed degree distribution.
    d_min: int
        The minimum degree considered in the Powerlaw distribution.
    Returns
    -------
    ll: tf.Tensor dtype float, (scalar)
        The log likelihood under the maximum likelihood estimate of the Powerlaw exponent alpha.
    alpha: tf.Tensor dtype float (scalar)
        The maximum likelihood estimate of the Powerlaw exponent.
    n: int
        The number of degrees in the degree sequence that are >= d_min.
    sum_log_degrees: tf.Tensor dtype float (scalar)
        The sum of the log of degrees in the distribution which are >= d_min.
    """
    # Determine which degrees are to be considered, i.e. >= d_min.
    in_range = tf.greater_equal(degree_sequence, d_min)
    # Sum the log of the degrees to be considered
    sum_log_degrees = tf.reduce_sum(tf.math.log(tf.boolean_mask(degree_sequence, in_range)))
    # Number of degrees >= d_min
    n = tf.cast(tf.math.count_nonzero(in_range), self.floatx)
    # Maximum likelihood estimate of the Powerlaw exponent
    alpha = compute_alpha(n, sum_log_degrees, d_min)
    # Log likelihood under alpha
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)

    return ll, alpha, n, sum_log_degrees


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    """
    Compute the change of the log likelihood of the Powerlaw distribution fit on the input adjacency matrix's degree
    distribution that results when adding/removing edges for the input node pairs. Assumes an undirected unweighted
    graph.
    Parameters
    ----------
    node_pairs: tf.Tensor, shape (e, 2) dtype int
        The e node pairs to consider, where each node pair consists of the two indices of the nodes.
    adjacency_matrix: tf.Tensor shape (N,N) dtype int
        The input adjacency matrix. Assumed to be unweighted and symmetric.
    d_min: int
        The minimum degree considered in the Powerlaw distribution.
    Returns
    -------
    new_ll: tf.Tensor of shape (e,) and dtype float
        The log likelihoods for node pair in node_pairs obtained when adding/removing the edge for that node pair.
    new_alpha: tf.Tensor of shape (e,) and dtype float
        For each node pair, contains the maximum likelihood estimates of the Powerlaw distributions obtained when
        adding/removing the edge for that node pair.
    new_n: tf.Tensor of shape (e,) and dtype float
        The updated number of degrees which are >= d_min for each potential edge being added/removed.
    sum_log_degrees_after: tf.Tensor of floats shape (e,)
        The updated sum of log degrees whose values are >= d_min for each of the e potential edges being added/removed.
    """

    # For each node pair find out whether there is an edge or not in the input adjacency matrix.
    edge_entries_before = tf.cast(tf.gather_nd(adjacency_matrix, tf.cast(node_pairs, self.intx)), self.floatx)
    # Compute the degree for each node
    degree_seq = tf.reduce_sum(adjacency_matrix, 1)

    # Determine which degrees are to be considered, i.e. >= d_min.
    in_range = tf.greater_equal(degree_seq, d_min)
    # Sum the log of the degrees to be considered
    sum_log_degrees = tf.reduce_sum(tf.math.log(tf.boolean_mask(degree_seq, in_range)))
    # Number of degrees >= d_min
    n = tf.cast(tf.math.count_nonzero(in_range), self.floatx)

    # The changes to the edge entries to add an edge if none was present and remove it otherwise.
    # i.e., deltas[ix] = -1 if edge_entries[ix] == 1 else 1
    deltas = -2 * edge_entries_before + 1

    # The degrees of the nodes in the input node pairs
    d_edges_before = tf.gather(degree_seq, tf.cast(node_pairs, self.intx))
    # The degrees of the nodes in the input node pairs after performing the change (i.e. adding the respective value of
    # delta.
    d_edges_after = tf.gather(degree_seq, tf.cast(node_pairs, self.intx)) + deltas[:, None]
    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Update the number of degrees >= d_min
    new_n = tf.cast(new_n, self.floatx)

    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)

    return new_ll, new_alpha, new_n, sum_log_degrees_after


def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    Parameters
    ----------
    node_pairs: tf.Tensor, shape (e, 2) dtype int
        The e node pairs to consider, where each node pair consists of the two indices of the nodes.
    modified_adjacency: tf.Tensor shape (N,N) dtype int
        The input (modified) adjacency matrix. Assumed to be unweighted and symmetric.
    original_adjacency: tf.Tensor shape (N,N) dtype int
        The input (original) adjacency matrix. Assumed to be unweighted and symmetric.
    d_min: int
        The minimum degree considered in the Powerlaw distribution.
    threshold: float, default 0.004
        Cutoff value for the unnoticeability constraint. Smaller means stricter constraint. 0.004 corresponds to a
        p-value of 0.95 in the Chi-square distribution with one degree of freedom.
    Returns
    -------
    allowed_mask: tf.Tensor, shape (e,), dtype bool
        For each node pair p return True if adding/removing the edge p does not violate the
        cutoff value, False otherwise.
    current_ratio: tf.Tensor, shape (), dtype float
        The current value of the log likelihood ratio.
    """

    N = int(modified_adjacency.shape[0])

    original_degree_sequence = tf.cast(tf.reduce_sum(original_adjacency, axis=1), self.floatx)
    current_degree_sequence = tf.cast(tf.reduce_sum(modified_adjacency, axis=1), self.floatx)

    # Concatenate the degree sequences
    concat_degree_sequence = tf.concat((current_degree_sequence[None, :], original_degree_sequence[None, :]), axis=1)
    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence,
                                                                                           d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)
    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence,
                                                                                           d_min)
    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.
    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               tf.cast(
                                                                                                   modified_adjacency,
                                                                                                   self.floatx), d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)
    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold
    filtered_edges = tf.boolean_mask(node_pairs, allowed_edges)

    # Get the flattened indices for the allowed edges [e,2] -> [e,], similar to np.ravel_multi_index
    flat_ixs = ravel_multiple_indices(tf.cast(filtered_edges, self.intx), modified_adjacency.shape)
    # Also for the reverse direction (we assume unweighted graphs).
    flat_ixs_reverse = ravel_multiple_indices(tf.reverse(tf.cast(filtered_edges, self.intx), [1]),
                                              modified_adjacency.shape)

    # Construct a [N * N] array with ones at the admissible node pair locations and 0 everywhere else.
    indices_1 = tf.scatter_nd(flat_ixs[:, None], tf.ones_like(flat_ixs, dtype=self.floatx), shape=[N * N])
    indices_2 = tf.scatter_nd(flat_ixs_reverse[:, None], tf.ones_like(flat_ixs_reverse, dtype=self.floatx),
                              shape=[N * N])

    # Add both directions
    allowed_mask = tf.clip_by_value(indices_1 + indices_2, 0, 1)

    allowed_mask = tf.reshape(allowed_mask, [N, N])
    return allowed_mask, current_ratio
