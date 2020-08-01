import numpy as np
import scipy.sparse as sp
from graphgallery import tqdm
from graphadv.defense.defender import Defender
from graphadv.utils import filter_singletons
from graphadv.utils.type_check import is_binary


class Jaccard(Defender):
    def __init__(self, adj, x, labels=None, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        if is_binary(x):
            self.is_binary = True
        else:
            self.is_binary = False

    def fit(self, threshold=0.01, disable=False):
        super().fit()
        if self.is_binary:
            similarity_fn = self._jaccard_similarity
        else:
            similarity_fn = self._cosine_similarity

        flips = []

        for row, col in tqdm(self.edges, desc='Preprocessing Graph', disable=disable):
            x_a = self.x[row]
            x_b = self.x[col]

            if similarity_fn(x_a, x_b) <= threshold:
                flips.append((row, col))

        if not self.allow_singleton and len(flips) > 0:
            flips = filter_singletons(flips, self.adj)

        self.threshold = threshold
        self.structure_flips = flips

    @staticmethod
    def _jaccard_similarity(a, b):
        intersection = np.count_nonzero(a*b)
        J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)
        return J

    @staticmethod
    def _cosine_similarity(a, b):
        inner_product = (a * b).sum()
        C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()))
        return C
