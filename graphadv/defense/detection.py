import warnings
import numpy as np
import scipy.sparse as sp

from graphadv.defense.defender import Defender
from graphadv.utils import filter_singletons
from graphadv.utils.type_check import is_binary
from graphadv import epsilon



class JaccardDetection(Defender):
    """ adj must be symmetric!
    
    """
    def __init__(self, adj, x, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x, seed=seed, name=name, device=device, **kwargs)
        
       
        
    def fit(self, threshold=0.01, disable=False):
        super().fit()
        x = self.x
        if not is_binary(x):
                warnings.warn(
                    "The input feature matrix is NOT binary! "
                    "For continuous features, you should use `graphadv.defense.CosinDetection` instead.",
                    RuntimeWarning,
                )             
            
        similarity_fn = self.jaccard_similarity
        rows, cols = sp.tril(self.adj, k=-1).nonzero()
        
        A = x[rows]
        B = x[cols]
        S = similarity_fn(A, B)    
        idx = np.where(S<=threshold)[0]
        flips = np.stack([rows[idx], cols[idx]], axis=1)

        if not self.allow_singleton and len(flips) > 0:
            flips = filter_singletons(flips, self.adj)

        self.threshold = threshold
        self.structure_flips = flips

    @staticmethod
    def jaccard_similarity(A, B):
        intersection = np.count_nonzero(A*B, axis=1)
        J = intersection * 1.0 / (np.count_nonzero(A, axis=1) + np.count_nonzero(B, axis=1) - intersection + epsilon())
        return J

class CosinDetection(Defender):
    """ adj must be symmetric!
    
    """
    def __init__(self, adj, x, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x, seed=seed, name=name, device=device, **kwargs)

    def fit(self, threshold=0.01, disable=False):
        super().fit()
        x = self.x
        if is_binary(x):
                warnings.warn(
                    "The input feature matrix is binary! "
                    "For binary features, you should use `graphadv.defense.JaccardDetection` instead.",
                    RuntimeWarning,
                )   
            
        similarity_fn = self.cosine_similarity
        rows, cols = sp.tril(self.adj, k=-1).nonzero()
        
        A = x[rows]
        B = x[cols]
        S = similarity_fn(A, B)    
        idx = np.where(S<=threshold)[0]
        flips = np.stack([rows[idx], cols[idx]], axis=1)

        if not self.allow_singleton and len(flips) > 0:
            flips = filter_singletons(flips, self.adj)

        self.threshold = threshold
        self.structure_flips = flips

    @staticmethod
    def cosine_similarity(A, B):
        inner_product = (A*B).sum(1)
        C = inner_product / (np.sqrt(np.square(A).sum(1)) * np.sqrt(np.square(B).sum(1)) + epsilon())
        return C

