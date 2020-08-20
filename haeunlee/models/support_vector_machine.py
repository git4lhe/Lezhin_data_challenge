from sklearn.svm import SVC


class SVM:
    @property
    def _model(self):
        return SVC()

    @property
    def _hyperparameters(self, kernel=None, C=None, gamma=None):
        # TODO: Add argument extension
        tuned_parameters = {
            "classifier__kernel": ["linear"],
            "classifier__gamma": [1e-3]
            # "model__kernel": ["linear", "rbf", "poly"],
            # "model__C": [0.1, 1],
            # "model__gamma": [1e-3, 1e-4, "auto", "scale"],
        }
        return tuned_parameters

    @property
    def train_settings(self):
        return self._model, self._hyperparameters
