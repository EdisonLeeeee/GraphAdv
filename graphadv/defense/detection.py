import warnings
import numpy as np
import scipy.sparse as sp

from graphadv.defense.defender import Defender
from graphadv.utils import filter_singletons
from graphadv.utils.type_check import is_binary
from graphadv import epsilon



def jaccard_similarity(A, B):
    intersection = np.count_nonzero(A*B, axis=1)
    J = intersection * 1.0 / (np.count_nonzero(A, axis=1) + np.count_nonzero(B, axis=1) - intersection + epsilon())
    return J
    
def cosine_similarity(A, B):
    inner_product = (A*B).sum(1)
    C = inner_product / (np.sqrt(np.square(A).sum(1)) * np.sqrt(np.square(B).sum(1)) + epsilon())
    return C

def filter_edges_by_similarity(adj, x, similarity_fn, threshold=0.01, allow_singleton=False):
    similarity_fn = jaccard_similarity
    rows, cols = sp.tril(adj, k=-1).nonzero()

    A = x[rows]
    B = x[cols]
    S = similarity_fn(A, B)    
    idx = np.where(S<=threshold)[0]
    flips = np.stack([rows[idx], cols[idx]], axis=1)
    if not allow_singleton and len(flips) > 0:
        flips = filter_singletons(flips, adj)
    return flips

def jaccard_detection(adj, x, threshold=0.01, allow_singleton=False):
    return filter_edges_by_similarity(adj, x, similarity_fn=jaccard_similarity,
                                      threshold=threshold, 
                                      allow_singleton=allow_singleton)

    
def cosin_detection(adj, x, threshold=0.01, allow_singleton=False):
    return filter_edges_by_similarity(adj, x, similarity_fn=cosine_similarity,
                                      threshold=threshold, 
                                      allow_singleton=allow_singleton)

class JaccardDetection(Defender):
    """ adj must be symmetric!
    
    """
    def __init__(self, adj, x, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x, seed=seed, name=name, device=device, **kwargs)
        if not is_binary(x):
            warnings.warn(
                "The input attribute matrix is NOT binary! "
                "For continuous attributes, you should use `graphadv.defense.CosinDetection` instead.",
                RuntimeWarning,
            )          
    def fit(self, threshold=0.01, disable=False):
        super().fit()

        self.threshold = threshold
        self.structure_flips = jaccard_detection(self.adj, self.x, 
                                                 threshold=threshold, 
                                                 allow_singleton=self.allow_singleton)

class CosinDetection(Defender):
    """ adj must be symmetric!
    
    """
    def __init__(self, adj, x, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x, seed=seed, name=name, device=device, **kwargs)
        if is_binary(x):
            warnings.warn(
                "The input attribute matrix is binary! "
                "For binary attributes, you should use `graphadv.defense.JaccardDetection` instead.",
                RuntimeWarning,
            )  
    def fit(self, threshold=0.01, disable=False):
        super().fit()
            
        self.threshold = threshold
        self.structure_flips = jaccard_detection(self.adj, self.x, 
                                                 threshold=threshold, 
                                                 allow_singleton=self.allow_singleton)


