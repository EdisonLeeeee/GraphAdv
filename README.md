# GraphAdv

<!-- [pypi-image]: https://badge.fury.io/py/graphadv.svg
[pypi-url]: https://pypi.org/project/graphadv/ -->
<!-- [![PyPI Version][pypi-image]][pypi-url] -->

<p align="center">
  <img width = "500" height = "400" src="https://github.com/EdisonLeeeee/GraphAdv/blob/master/imgs/graphadv.svg" alt="logo"/>
</p>

---

[![Python 3.6](https://img.shields.io/badge/Python->=3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow >=2.1](https://img.shields.io/badge/TensorFlow->=2.1-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0)
[![PyPI version](https://badge.fury.io/py/graphadv.svg)](https://badge.fury.io/py/graphadv)
![](https://img.shields.io/github/forks/EdisonLeeeee/GraphAdv)
![](https://img.shields.io/github/stars/EdisonLeeeee/GraphAdv)
![](https://img.shields.io/github/issues/EdisonLeeeee/GraphAdv)
[![GitHub license](https://img.shields.io/github/license/EdisonLeeeee/GraphAdv)](https://github.com/EdisonLeeeee/GraphAdv/blob/master/LICENSE)

TensorFlow 2 implementation of state-of-the-arts graph adversarial attack and defense models (methods). This repo is built on another graph-based repository [GraphGallery](https://github.com/EdisonLeeeee/GraphGallery), You can browse it for more details.

# Installation
```bash
# graphgallery is necessary for this package
pip install -U graphgallery
pip install -U graphadv
```

# Quick Start
+ Targeted Attack
```python
from graphadv.attack.targeted import Nettack
attacker = Nettack(adj, x, labels, idx_train, seed=None)
# reset for next attack
attacker.reset()
# By default, the number of perturbations is set to the degree of nodes, you can change it by `n_perturbations=`
attacker.attack(target, direct_attack=True, structure_attack=True, feature_attack=False)

# get the edge flips
>>> attacker.edge_flips
# get the attribute flips
>>> attacker.attr_flips
# get the perturbed adjacency matrix
>>> attacker.A
# get the perturbed attribute matrix
>>> attacker.X

```
+ Untargeted Attack
```python
from graphadv.attack.untargeted import Metattack
attacker = Metattack(adj, x, labels, 
                     idx_train, idx_unlabeled=idx_unlabeled, 
                     lr=0.01, # cora and cora_ml lr=0.1 citeseer lr=0.01
                     lambda_=1.0,
                     device="GPU", seed=None)
# reset for next attack
attacker.reset()
# By default, the number of perturbations is set to the degree of nodes, you can change it by `n_perturbations=`
# `n_perturbations` can be integer (number of edges) or float scalar (>=0, <=1, the ratio of edges)
attacker.attack(0.05, structure_attack=True, feature_attack=False)

# get the edge flips
>>> attacker.edge_flips
# get the attribute flips
>>> attacker.attr_flips
# get the perturbed adjacency matrix
>>> attacker.A
# get the perturbed attribute matrix
>>> attacker.X

```
+ Defense
+ `JaccardDetection` for binary node attributes
+ `CosinDetection` for continuous node attributes

```python
from graphadv.defense import JaccardDetection, CosinDetection
defender = JaccardDetection(adj, x)
defender.reset()
defender.fit()

# get the modified adjacency matrix
>>> defender.A
# get the modified attribute matrix
>>> defender.X
```
More examples please refer to the [examples](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples) directory.


# Acknowledgement
This project is motivated by [DeepRobust](https://github.com/DSE-MSU/DeepRobust), and the original implementations of the authors, thanks for their excellent works!


