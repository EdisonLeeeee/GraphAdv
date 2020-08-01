import numpy as np
from graphgallery.nn.models import (GCN, SGC, GAT, ClusterGCN, RobustGCN, GWNN,
                                    GraphSAGE, GCN_MIX, FastGCN, ChebyNet, DenseGCN,
                                    Deepwalk, Node2vec)


def evaluate(adj, x, labels, idx_train, idx_val, idx_test, target, retrain_iters=2, norm_x='l1'):

    classification_margins = []
    class_distrs = []
    for _ in range(retrain_iters):
        print(f"... {_+1}/{retrain_iters}")
        model = GCN(adj, x, labels, device='GPU:0', seed=123+_, norm_x=norm_x)
        model.build()
        his = model.train(idx_train, idx_val, verbose=0, epochs=100)
        logit = model.predict(target).ravel()

        class_distrs.append(logit)
        best_second_class_before = (logit - np.eye(data.n_classes)[labels[target]]).argmax()
        margin = logit[labels[target]] - logit[best_second_class_before]
        classification_margins.append(margin)
        model.close

    class_distrs = np.asarray(class_distrs)
    print(classification_margins)
    return class_distrs
