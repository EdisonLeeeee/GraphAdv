import numpy as np
import scipy.sparse as sp
from graphadv import BaseModel


class Defender(BaseModel):
    def __init__(self, adj, x=None, labels=None, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

    def reset(self):
        self.modified_adj = None
        self.modified_x = None
        self.modified_degree = None
        self.feature_flips = None
        self.structure_flips = None
        self.reseted = True

    def fit(self):
        if not self.reseted:
            raise RuntimeError('Before calling attack once again, you must reset your model. Use `model.reset()`.')
        self.reseted = False

    def __repr__(self):
        return 'Defender: ' + self.name + ' in ' + self.device
