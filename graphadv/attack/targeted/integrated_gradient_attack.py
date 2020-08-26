import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from graphadv import is_binary
from graphadv.attack.targeted.targeted_attacker import TargetedAttacker
from graphadv.utils import train_a_surrogate, largest_indices, filter_singletons
from graphgallery.nn.models import DenseGCN
from graphgallery import tqdm, astensor, normalize_adj_tensor



class IG(TargetedAttacker):
    def __init__(self, adj, x, labels, idx_train=None,
                 seed=None, name=None, device='CPU:0', surrogate=None, surrogate_args={}, **kwargs):
        super().__init__(adj, x=x, labels=labels, seed=seed, name=name, device=device, **kwargs)

        if surrogate is None:
            surrogate = train_a_surrogate(self, 'DenseGCN', idx_train, **surrogate_args)
        elif not isinstance(surrogate, DenseGCN):
            raise RuntimeError("surrogate model should be the instance of `graphgallery.nn.DenseGCN`.")

        adj, x = self.adj, self.x
        self.nodes_set = set(range(self.n_nodes))
        self.features_set = np.arange(self.n_features)
        
        # IG can also conduct feature attack
        self.allow_feature_attack = True
        
        with tf.device(self.device):
            self.surrogate = surrogate
            self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
            self.tf_x = astensor(x)    
            self.tf_adj = astensor(adj.A)
            self.adj_norm = normalize_adj_tensor(self.tf_adj)

    def reset(self):
        super().reset()

        
    def attack(self, target, n_perturbations=None, direct_attack=True,
               structure_attack=True, feature_attack=False, steps=10, disable=False):
        
        super().attack(target, n_perturbations, direct_attack, structure_attack, feature_attack)
        
        if feature_attack and not is_binary(self.x):
            raise ValueError("Attacks on the node features are currently only supported for binary attributes.")
            
        if structure_attack:
            candidate_edges = self.get_candidate_edges()
            link_importance = self.get_link_importance(candidate_edges, steps)
        
        if feature_attack:
            candidate_features = self.get_candidate_features()
            feature_importance = self.get_feature_importance(candidate_features, steps)

        if structure_attack and not feature_attack:
            self.structure_flips = candidate_edges[largest_indices(link_importance, self.n_perturbations)[0]]
        elif feature_attack and not structure_attack:
            self.attribute_flips = candidate_features[largest_indices(feature_importance, self.n_perturbations)[0]]
        else:
            # both attacks are conducted
            link_selected = []
            feature_selected = []            
            importance = np.hstack([link_importance, feature_importance])
            boundary = link_importance.size
            
            for index in largest_indices(importance, self.n_perturbations)[0]:
                if index<boundary:
                    link_selected.append(index)
                else:
                    feature_selected.append(index-boundary)
                    
            if link_selected:
                self.structure_flips = candidate_edges[link_selected]
            if feature_selected:
                self.attribute_flips = candidate_features[feature_selected]
                
            
    def get_candidate_edges(self):
        n_nodes = self.n_nodes
        target = self.target
        
        if self.direct_attack:
            influence_nodes = [target]
            candidate_edges = np.column_stack(
                (np.tile(target, n_nodes-1), list(self.nodes_set-set([target]))))
        else:
            influence_nodes = self.adj[target].nonzero()[1]
            candidate_edges = np.row_stack([np.column_stack((np.tile(infl, n_nodes - 2),
                                                             list(self.nodes_set - set([target, infl])))) for infl in
                                            influence_nodes])
            
        if not self.allow_singleton:
            candidate_edges = filter_singletons(candidate_edges, self.adj)            
            
        return candidate_edges
    
    def get_candidate_features(self):
        n_features = self.n_features
        target = self.target
        
        if self.direct_attack:
            influence_nodes = [target]
            candidate_features = np.column_stack((np.tile(target, n_features), self.features_set))
        else:
            influence_nodes = self.adj[target].nonzero()[1]
            candidate_features = np.row_stack([np.column_stack((np.tile(infl, n_features), self.features_set)) for infl in influence_nodes])
            
        return candidate_features
            
    def get_link_importance(self, candidates, steps=10):
        
        adj = self.tf_adj
        x = self.tf_x
        target_index = astensor([self.target])
        target_label = astensor(self.target_label)
        baseline_add = adj.numpy()
        baseline_add[candidates[:,0], candidates[:,1]] = 1.0
        baseline_add = astensor(baseline_add)
        baseline_remove = adj.numpy()
        baseline_remove[candidates[:,0], candidates[:,1]] = 0.0
        baseline_remove = astensor(baseline_remove)
        edge_indicator = self.adj[candidates[:,0], candidates[:,1]].A1 > 0
        
        edges = candidates[edge_indicator]
        non_edges = candidates[~edge_indicator]
        
        edge_gradients = tf.zeros(edges.shape[0])
        non_edge_gradients = tf.zeros(non_edges.shape[0])
        
        for alpha in tf.linspace(0., 1.0, steps+1):
            ###### Compute integrated gradients for removing edges ######
            adj_diff = adj - baseline_remove
            adj_step = baseline_remove + alpha * adj_diff
            
            gradients = self.compute_structure_gradients(adj_step, x, target_index, target_label)
            edge_gradients += -tf.gather_nd(gradients, edges)
            
            ###### Compute integrated gradients for adding edges ######
            adj_diff = baseline_add - adj
            adj_step = baseline_add - alpha * adj_diff            
                  
            gradients = self.compute_structure_gradients(adj_step, x, target_index, target_label)
            non_edge_gradients += tf.gather_nd(gradients, non_edges) 

        integrated_grads = np.zeros(edge_indicator.size)
        integrated_grads[edge_indicator] = edge_gradients.numpy()
        integrated_grads[~edge_indicator] = non_edge_gradients.numpy()
        
        return integrated_grads
    
    def get_feature_importance(self, candidates, steps=10):
        adj = self.adj_norm
        x = self.tf_x
        target_index = astensor([self.target])
        target_label = astensor(self.target_label)
        baseline_add = x.numpy()
        baseline_add[candidates[:,0], candidates[:,1]] = 1.0
        baseline_add = astensor(baseline_add)
        baseline_remove = x.numpy()
        baseline_remove[candidates[:,0], candidates[:,1]] = 0.0
        baseline_remove = astensor(baseline_remove)
        feature_indicator = self.x[candidates[:,0], candidates[:,1]] > 0
        
        features = candidates[feature_indicator]
        non_features = candidates[~feature_indicator]
        
        feature_gradients = tf.zeros(features.shape[0])
        non_feature_gradients = tf.zeros(non_features.shape[0])
        
        for alpha in tf.linspace(0., 1.0, steps+1):
            ###### Compute integrated gradients for removing features ######
            x_diff = x - baseline_remove
            x_step = baseline_remove + alpha * x_diff
            
            gradients = self.compute_feature_gradients(adj, x_step, target_index, target_label)
            feature_gradients += -tf.gather_nd(gradients, features)
            
            ###### Compute integrated gradients for adding features ######
            x_diff = baseline_add - x
            x_step = baseline_add - alpha * x_diff            
                  
            gradients = self.compute_feature_gradients(adj, x_step, target_index, target_label)
            non_feature_gradients += tf.gather_nd(gradients, non_features) 

        integrated_grads = np.zeros(feature_indicator.size)
        integrated_grads[feature_indicator] = feature_gradients.numpy()
        integrated_grads[~feature_indicator] = non_feature_gradients.numpy()
        
        return integrated_grads
    
    @tf.function
    def compute_structure_gradients(self, adj, x, target_index, target_label):
        
        with tf.GradientTape() as tape:
            tape.watch(adj)
            adj_norm = normalize_adj_tensor(adj)            
            logit = self.surrogate([x, adj_norm, target_index])
            loss = self.loss_fn(target_label, logit) 
            
        gradients = tape.gradient(loss, adj)
        return gradients
    
    
    @tf.function
    def compute_feature_gradients(self, adj, x, target_index, target_label):
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            logit = self.surrogate([x, adj, target_index])
            loss = self.loss_fn(target_label, logit) 
            
        gradients = tape.gradient(loss, x)
        return gradients    

