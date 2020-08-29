# Implementations
In detail, the following methods are currently implemented:

## Attack
### Targeted Attack
+ **RAND**: The the simplest attack method.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Targeted%20Attack/test_RAND.ipynb)
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
+ **RAND**: The the simplest attack method.
[[🌈 Example]](https://github.com/EdisonLeeeee/GraphAdv/tree/master/examples/Untargeted%20Attack/test_RAND.ipynb)
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