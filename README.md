# GraphAdv

<!-- <p align="center">
  <img width = "500" height = "250" src="https://github.com/EdisonLeeeee/GraphAdv/blob/master/imgs/graphadv.svg" alt="logo"/>
</p>

--- -->

[![Python 3.6](https://img.shields.io/badge/Python->=3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow >=2.1](https://img.shields.io/badge/TensorFlow->=2.1-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0)
[![PyPI version](https://badge.fury.io/py/graphadv.svg)](https://badge.fury.io/py/graphadv)
[![GitHub license](https://img.shields.io/github/license/EdisonLeeeee/GraphAdv)](https://github.com/EdisonLeeeee/GraphAdv/blob/master/LICENSE)

# NOTE: GraphAdv is still Building due the upgrading of GraphGallery

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

# Implementations
In detail, the following methods are currently implemented:

## Attack
### Targeted Attack
+ **RAND**: The simplest attack method.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_RA.ipynb)
+ **FGSM**, from *Ian J. Goodfellow et al.*, [📝Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), *ICLR'15*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_FGSM.ipynb)
+ **DICE**, from *Marcin Waniek et al*, [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior 16*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_DICE.ipynb)
+ **Nettack**, from *Daniel Zügner et al.*, [📝Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984), *KDD'18*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_Nettack.ipynb)
+ **IG**, from *Huijun Wu et al.*, [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_IG.ipynb)
+ **GF-Attack**, from *Heng Chang et al*, [📝A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models](https://arxiv.org/abs/1908.01297), *AAAI'20*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_GFA.ipynb)
+ **IGA**, from *Jinyin Chen et al.*, [📝Link Prediction Adversarial Attack Via Iterative Gradient Attack](https://ieeexplore.ieee.org/abstract/document/9141291), *IEEE Trans 20*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_IGA.ipynb)
+ **SGA**, from.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_SGA.ipynb)

### Untargeted Attack
+ **RAND**: The simplest attack method.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted%20Attack/test_RA.ipynb)
+ **FGSM**, from *Ian J. Goodfellow et al.*, [📝Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), *ICLR'15*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted%20Attack/test_FGSM.ipynb)
+ **DICE**, from *Marcin Waniek et al*, [📝Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375), *Nature Human Behavior 16*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted%20Attack/test_DICE.ipynb)
+ **Metattack**, **MetaApprox**, from *Daniel Zügner et al.*, [📝Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412), *ICLR'19*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted%20Attack/test_Metattack.ipynb), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted%20Attack/test_MetaApprox.ipynb)
+ **Degree**, **Node Embedding Attack**, from *Aleksandar Bojchevski et al.*, [📝Adversarial Attacks on Node Embeddings via Graph Poisoning](https://arxiv.org/abs/1809.01093), *ICLR'19*.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted%20Attack/test_Degree.ipynb), [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted%20Attack/test_node_embedding_attack.ipynb)
+ **PGD**, **MinMax**, from *Kaidi Xu et al.*, [📝Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19*.
[[🌈 Poisoning Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted%20Attack/test_PGD_poisoning.ipynb), [[🌈 Poisoning Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted%20Attack/test_MinMax_poisoning.ipynb), [[🌈 Evasion Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Untargeted%20Attack/test_PGD_evasion.ipynb)

## Defense
+ **JaccardDetection**, **CosinDetection**, from *Huijun Wu et al.*, [📝Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610), *IJCAI'19*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/test_detection.ipynb)
+ **Adversarial Tranining**, from *Kaidi Xu et al.*, [📝Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214), *IJCAI'19*.
+ **SVD**, from *Negin Entezari et al.*, [📝All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/abs/10.1145/3336191.3371789), *WSDM'20*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/test_svd.ipynb)
+ **RGCN**, from *Dingyuan Zhu et al.*, [Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf), *KDD'19*.
 [[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/blob/master/examples/Defense/test_RGCN.ipynb)

More details of the official papers and codes can be found in [Awesome Graph Adversarial Learning](https://github.com/gitgiter/Graph-Adversarial-Learning).


# Acknowledgement
This project is motivated by [DeepRobust](https://github.com/DSE-MSU/DeepRobust), and the original implementations of the authors, thanks for their excellent works!


