import abc
import numpy as np

from graphadv.base import BaseModel
from graphadv.utils.type_check import is_scalar_like


class BaseAttacker(BaseModel):

    def __init__(self, adj, x=None, labels=None, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)
        
        self.__max_perturbations = 0
        self.__allow_feature_attack = False
        self.__allow_structure_attack = True


    @abc.abstractmethod
    def attack(self):
        '''for attack model.'''
        raise NotImplementedError

    def _check_budget(self, n_perturbations, max_perturbations):
        max_perturbations = max(max_perturbations, self.__max_perturbations)
        
        if not is_scalar_like(n_perturbations) or n_perturbations <= 0:
            raise ValueError(f'`n_perturbations` must be a postive integer scalar. but got `{n_perturbations}`.')

        if n_perturbations > max_perturbations:
            raise ValueError(f'`n_perturbations` should be less than or equal the maximum allowed perturbations: {max_perturbations}.')

        if n_perturbations < 1.:
            n_perturbations = max_perturbations * n_perturbations

        return int(n_perturbations)


    @property
    def allow_structure_attack(self):
        return self.__allow_structure_attack

    @allow_structure_attack.setter
    def allow_structure_attack(self, value):
        self.__allow_structure_attack = value

    @property
    def allow_feature_attack(self):
        return self.__allow_feature_attack

    @allow_feature_attack.setter
    def allow_feature_attack(self, value):
        self.__allow_feature_attack = value

    def set_max_perturbations(self, max_perturbations=np.inf, verbose=True):
        assert is_scalar_like(max_perturbations)
        self.__max_perturbations = max_perturbations
        if verbose:
            print(f"max_perturbations: {max_perturbations}")

    def __repr__(self):
        return 'Attacker: ' + self.name + ' in ' + self.device
