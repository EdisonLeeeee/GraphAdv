import numpy as np

from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphgallery import tqdm


class Target2General(UntargetedAttacker):
    def __init__(self, adj, x, labels, attacker, seed=None, name=None, device='CPU:0', **kwargs):
        super().__init__(adj=adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        attacker.set_max_perturbations(vervose=False)
        self.attacker = attacker

    def attack(self, index_attack, n_perturbations=0.05,
               structure_attack=True, feature_attack=False, disable=False):

        super().attack(n_perturbations, structure_attack, feature_attack)
        total_perturbations = self.n_perturbations
        attacker = self.attacker

        degree = self.degree[index_attack].astype(self.intx)
        if degree.sum() < total_perturbations:
            raise RuntimeError('The degree sum of attacked nodes is less than the number of perturbations! Please add more nodes to attack.')
        sorted_index = np.argsort(degree)
        index_attack = index_attack[sorted_index]
        degree = degree[sorted_index]

        flips = []

        with tqdm(total=total_perturbations, desc='Peturbing Graph', disable=disable) as pbar:

            for deg, target in zip(degree, index_attack):
                single_node_perturbations = min(deg, total_perturbations)
                attacker.reset()
                attacker.attack(target, single_node_perturbations, disable=True)
                attacker.set_adj(attacker.A)
                flips.append(attacker.edge_flips)
                total_perturbations -= single_node_perturbations
                pbar.update(single_node_perturbations)
                if total_perturbations == 0:
                    break
                    
        self.structure_flips = np.vstack(flips)
