import abc
import random
import warnings
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from graphadv.utils.type_check import (is_scalar_like,
                                       is_tensor_or_variable, is_symmetric,
                                       is_binary, is_self_loops,
                                       check_and_convert)
from graphadv import floatx, intx
from graphadv import flip_adj, flip_x


class BaseModel:

    def __init__(self, adj, x=None, labels=None,
                 seed=None, name=None, device='CPU:0', **kwargs):

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if name is None:
            name = self.__class__.__name__

        # check if some invalid arguments
        allowed_kwargs = set([])
        unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
        if unknown_kwargs:
            raise ValueError(
                "Invalid keyword argument(s) in `__init__`: %s" % (unknown_kwargs,))

        if any((not is_binary(adj), not is_symmetric(adj), is_self_loops(adj))):
            raise ValueError('The input adjacency matrix should be symmertic, unweighted and without self loops.')

        adj = check_and_convert(adj, is_sparse=True)
        if x is not None:
            x = check_and_convert(x, is_sparse=False)

        self.floatx = floatx()
        self.intx = intx()

        self.adj = adj
        self.x = x
        self.labels = labels
        self.degree = (adj.sum(1).A1 - adj.diagonal()).astype(self.intx)

        self.n_nodes = adj.shape[0]
        self.n_edges = adj.nnz//2

        if labels is not None:
            self.n_classes = labels.max() + 1
        else:
            self.n_classes = None

        if x is not None:
            self.n_attrs = x.shape[1]
        else:
            self.n_attrs = None

        self.seed = seed
        self.name = name
        self.device = device
        self.feature_flips = None
        self.structure_flips = None

        self.__reseted = False
        self.__allow_singleton = False

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @property
    def A(self):
        edge_flips = self.edge_flips
        if self.modified_adj is None:
            if edge_flips is not None:
                self.modified_adj = flip_adj(self.adj, edge_flips)
            else:
                self.modified_adj = self.adj

        adj = self.modified_adj

        if is_tensor_or_variable(adj):
            adj = adj.numpy()

        if isinstance(adj, (np.ndarray, np.matrix)):
            adj = sp.csr_matrix(adj)
        elif sp.isspmatrix(adj):
            adj = adj.tocsr(copy=False)
        else:
            raise TypeError(f'Invalid type of `modified_adj`, type {type(adj)}, please check it again.')

        return adj

    @property
    def X(self):
        attr_flips = self.attr_flips
        if self.modified_x is None and attr_flips is not None:
            self.modified_x = flip_x(self.x, attr_flips)

        if attr_flips is None:
            x = self.x
        else:
            x = self.modified_x
        if sp.isspmatrix(x):
            x = x.A
        elif is_tensor_or_variable(x):
            x = x.numpy()
        return x

    @property
    def D(self):
        if self.modified_degree is None:
            self.modified_degree = self.A.sum(1).A1.astype(self.intx)
        return self.modified_degree

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def attack(self):
        '''for attack model.'''
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        '''for defense method'''
        raise NotImplementedError

    @property
    def edge_flips(self):

        if isinstance(self.structure_flips, (list, np.ndarray)):
            return np.asarray(self.structure_flips)
        elif isinstance(self.structure_flips, dict):
            return np.asarray(list(self.structure_flips.keys()))
        elif self.structure_flips is None:
            return None
        else:
            raise TypeError(f'Invalid type of `structure_flips`, type `{self.structure_flips}`.')

    @property
    def attr_flips(self):

        if isinstance(self.feature_flips, (list, np.ndarray)):
            return np.asarray(self.feature_flips)
        elif isinstance(self.feature_flips, dict):
            return np.asarray(list(self.feature_flips.keys()))
        elif self.feature_flips is None:
            return None
        else:
            raise TypeError(f'Invalid type of `feature_flips`, type `{self.feature_flips}`.')

    @property
    def allow_singleton(self):
        return self.__allow_singleton

    @allow_singleton.setter
    def allow_singleton(self, value):
        self.__allow_singleton = value

    @property
    def reseted(self):
        return self.__reseted

    @reseted.setter
    def reseted(self, value):
        self.__reseted = value

    def set_adj(self, adj):
        '''Reset the adjacency matrix'''
        if any((not is_binary(adj), not is_symmetric(adj, is_self_loops(adj)))):
            raise ValueError('The input adjacency matrix should be symmertic, unweighted and without self loops.')

        adj = check_and_convert(adj, is_sparse=True)
        self.adj = adj
        self.degree = (adj.sum(1).A1 - adj.diagonal()).astype(self.intx)
        self.n_nodes = adj.shape[0]
        self.n_edges = adj.nnz//2

    def set_x(self, x):
        '''Reset the feature matrix'''
        x = check_and_convert(x, is_sparse=False)
        x_shape = x.shape
        assert x_shape[0] == self.n_nodes
        self.x = x
        self.n_attrs = x_shape[1]

    def show_edge_flips(self, detail=False):
        assert self.labels is not None
        flips = self.edge_flips
        _add = _del = 0
        add_and_diff = del_and_same = 0

        for u, v in flips:
            class_u = self.labels[u]
            class_v = self.labels[v]
            if self.adj[u, v] > 0:
                _del += 1
                if class_u == class_v:
                    del_and_same += 1
                if detail:
                    print(f'Del an edge ({u:<3} <-> {v:>3}), class {u:<3}= {class_u}, class {v:<3}= {class_v}.')
            else:
                _add += 1
                if class_u != class_v:
                    add_and_diff += 1
                if detail:
                    print(f'Add an edge ({u:<3} <-> {v:>3}), class {u:<3}= {class_u}, class {v:<3}= {class_v}.')

        print(f'Flip {_add+_del} edges, {_add} added, {_del} removed. Added edges with different classes: {add_and_diff/_add if _add > 0. else 0.:.2%}, removed edges with the same classes: {del_and_same/_del if _del > 0. else 0.:.2%}')

    def __repr__(self):
        return self.name + ' in ' + self.device
