from tensorflow.keras.optimizers import Adam

from graphgallery import tqdm
from graphgallery.nn.models import SemiSupervisedModel
from graphadv.attack.untargeted.untargeted_attacker import UntargetedAttacker



class AdvTraining:
    '''
        Adversarial training Framework for defending against UNtargeted attacks
    '''

    def __init__(self, model, attacker):
        assert isinstance(model, SemiSupervisedModel), 'Surrogate model should be the instance of `graphgallery.SemiSupervisedModel`.'

        assert isinstance(attacker, UntargetedAttacker), 'The attack model must be the instance of `graphadv.UntargetedAttacker`.'

        self.model = model
        self.attacker = attacker

    def fit(self, epochs=10, n_perturbations=0.05, lr=0.001, attacker_kwargs={}, model_kwargs={}, disable=False):

        model = self.model
        attacker = self.attacker
        assert attacker.x is not None, 'You must specify the feature matrix `x` for attacker to fed into classifiers.'

        if model.idx_test is not None:
            loss, accuracy = self.test(n_perturbations=n_perturbations,
                                       attacker_kwargs=attacker_kwargs)
            print(f'Accuracy before adversarial training: {accuracy}.')

        model.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
        for _ in tqdm(range(epochs), desc='Adversarial Training', disable=disable):
            attacker.reset()
            attacker.attack(n_perturbations, **attacker_kwargs, disable=True)
            model.preprocess(attacker.A, attacker.X)
            model.train(model.idx_train, save_best=False, validation=False, **model_kwargs)

        if model.idx_test is not None:
            loss, accuracy = self.test(n_perturbations=n_perturbations,
                                       attacker_kwargs=attacker_kwargs)

            print(f'Accuracy after adversarial training: {accuracy}.')

    def test(self, n_perturbations, attacker_kwargs):

        model = self.model
        attacker = self.attacker
        attacker.reset()
        attacker.attack(n_perturbations, disable=True, **attacker_kwargs)
        model.preprocess(attacker.A, attacker.X)
        loss, accuracy = model.test(model.idx_test)
        return loss, accuracy
