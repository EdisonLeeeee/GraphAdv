from graphgallery.nn.models import SGC, GCN, DenseGCN


def train_a_surrogate(attacker, surrogate, idx_train, idx_val=None, **kwargs):
    if idx_train is None:
        raise RuntimeError('You must specified the `idx_train` to train a surrogate model.')
        
    allowed_surrogate = ('SGC', 'GCN', 'DenseGCN')
    if surrogate not in allowed_surrogate:
        raise ValueError(f'Invalid surrogate model `{surrogate}`, allowed surrogate models: `{allowed_surrogate}`.')

    print(f'=== {attacker.name}: Train a surrogate model `{surrogate}` from scratch ===')
    model = eval(surrogate)(attacker.adj, attacker.x, attacker.labels, seed=attacker.seed,
                            norm_x=kwargs.pop('norm_x', None),
                            device=kwargs.pop('device', attacker.device))
    if surrogate!= 'SGC':
        model.build(hiddens=kwargs.pop('hiddens', 16), activations=kwargs.pop('activations', 'relu'))
    else:
        model.build()
        
    his = model.train(idx_train, idx_val, verbose=0,
                      epochs=kwargs.pop('epochs', 200),
                      save_best=kwargs.pop('save_best', True))

    return model
