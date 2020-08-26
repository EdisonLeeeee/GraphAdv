import numpy as np
import graphgallery
from graphadv.attack.baseattacker import BaseAttacker


class TargetedAttacker(BaseAttacker):
    def __init__(self, adj, x=None, labels=None, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

    def reset(self):
        self.modified_adj = None
        self.modified_x = None
        self.modified_degree = None
        self.target = None
        self.target_label = None
        self.feature_flips = None
        self.structure_flips = None
        self.n_perturbations = None
        self.structure_attack = None
        self.feature_attack = None
        self.direct_attack = None
        self.reseted = True

    def attack(self, target, n_perturbations, direct_attack, structure_attack, feature_attack):

        if not self.reseted:
            raise RuntimeError('Before calling attack once again, you must reset your attacker. Use `attacker.reset()`.')

        if not graphgallery.is_interger_scalar(target):
            raise ValueError('The `target` must be the instance of `int`, and only support attacking one node each time.')

        if not (structure_attack or feature_attack):
            raise RuntimeError('Either `structure_attack` or `feature_attack` must be True.')

        if feature_attack and not self.allow_feature_attack:
            raise RuntimeError(f"{self.name} does NOT support attacking features."
                               " If the model can conduct feature attack, please call `attacker.allow_feature_attack=True`.")

        if structure_attack and not self.allow_structure_attack:
            raise RuntimeError(f"{self.name} does NOT support attacking structures."
                               " If the model can conduct structure attack, please call `attacker.allow_structure_attack=True`.")

        if n_perturbations is None:
            n_perturbations = self.degree[target]
        else:
            n_perturbations = self._check_budget(n_perturbations, max_perturbations=self.degree[target])

        self.target = target
        self.target_label = self.labels[target] if self.labels is not None else None
        self.n_perturbations = n_perturbations
        self.direct_attack = direct_attack
        self.structure_attack = structure_attack
        self.feature_attack = feature_attack
        self.reseted = False

    def is_modified_edge(self, u, v):
        if self.direct_attack:
            return any((u == v,
                        self.target not in (u, v),
                        (u, v) in self.structure_flips,
                        (v, u) in self.structure_flips))
        else:
            return any((u == v,
                        self.target in (u, v),
                        (u, v) in self.structure_flips,
                        (v, u) in self.structure_flips))
