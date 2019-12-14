from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.utils import check_random_state


class Learner:
    """
    Class containing multiple sklearn models
    """

    def __init__(self, name="", rng=None):
        self._model = None
        self.name = name
        self.rng = check_random_state(rng)

    def get_params(self, deep=True):
        return self._model.get_params()

    def fit(self, X, y):
        self._model.fit(X, y)

    def decision_function(self, X):
        return self._model.decision_function(X)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class SVM(Learner):
    def __init__(self, name='svm', rng=None, gamma='scale', C=1):
        super().__init__(name, rng)

        model = None
        if name == 'svm':
            model = LinearSVC(
                penalty='l2',
                loss='hinge',
                multi_class='ovr',
                random_state=self.rng)

        elif name == 'svm_rbf':
            model = SVC(gamma=gamma, C=C, random_state=self.rng)

        elif name == "default":
            model = SVC()

        elif name == 'l1svm':
            model = LinearSVC(
                penalty='l1',
                loss='squared_hinge',
                dual=False,
                multi_class='ovr',
                random_state=self.rng)

        self._model = model


class GNB(Learner):
    def __init__(self, name="Gaussian Naive Bayes", rng=None):
        super().__init__(name, rng)

        self._model = GaussianNB()


class NeuralNetwork(Learner):
    def __init__(self, name="MLPClassifier", rng=None):
        super().__init__(name, rng)

        self._model = MLPClassifier()


class LogRegression(Learner):
    def __init__(self, name='Logistic Regression', rng=None):
        super().__init__(name, rng)

        self._model = LogisticRegression(
            penalty='l2',
            multi_class='ovr',
            fit_intercept=False,
            random_state=self.rng)


class GradientBoosting(Learner):
    def __init__(self, name="Gradient Boosting", rng=None):
        super().__init__(name, rng)

        kwargs = {
            'n_estimators': 1200,
            'max_depth': 3,
            'subsample': 0.5,
            'learning_rate': 0.01,
            'min_samples_leaf': 1,
            'random_state': self.rng,
        }
        self._model = GradientBoostingClassifier(**kwargs)

