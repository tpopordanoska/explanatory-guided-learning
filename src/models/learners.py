from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
        self.sklearn_model = None
        self.name = name
        self.rng = check_random_state(rng)

    def get_params(self, deep=True):
        return self.sklearn_model.get_params()

    def fit(self, X, y, sample_weight):
        self.sklearn_model.fit(X, y, sample_weight=sample_weight)

    def decision_function(self, X):
        return self.sklearn_model.decision_function(X)

    def predict(self, X):
        return self.sklearn_model.predict(X)

    def predict_proba(self, X):
        return self.sklearn_model.predict_proba(X)


class SVM(Learner):
    def __init__(self, name='svm', rng=None, gamma='scale', C=1):
        super().__init__(name, rng)

        if name == 'svm':
            model = LinearSVC(
                penalty='l2',
                loss='hinge',
                multi_class='ovr',
                random_state=self.rng)

        elif name == "default":
            model = SVC(random_state=self.rng)

        elif name == 'l1svm':
            model = LinearSVC(
                penalty='l1',
                loss='squared_hinge',
                dual=False,
                multi_class='ovr',
                random_state=self.rng)

        else:
            model = SVC(gamma=gamma, C=C, random_state=self.rng, probability=True)

        self.sklearn_model = model


class GNB(Learner):
    def __init__(self, name="Gaussian Naive Bayes", rng=None):
        super().__init__(name, rng)

        self.sklearn_model = GaussianNB()


class RandomForrest(Learner):
    def __init__(self, name="Random Forrest Classifier", n_estimators=100, max_features="auto", max_depth=None, rng=None):
        super().__init__(name, rng)

        self.sklearn_model = RandomForestClassifier(n_estimators=n_estimators,
                                                    max_features=max_features,
                                                    max_depth=max_depth,
                                                    random_state=rng)


class NeuralNetwork(Learner):
    def __init__(self, name="Multilayer Perceptron", rng=None):
        super().__init__(name, rng)

        self.sklearn_model = MLPClassifier(random_state=rng, alpha=0.1)


class LogRegression(Learner):
    def __init__(self, name='Logistic Regression', rng=None):
        super().__init__(name, rng)

        self.sklearn_model = LogisticRegression(random_state=rng)


class GradientBoosting(Learner):
    def __init__(self, name="Gradient Boosting Classifier", rng=None, n_estimators=100):
        super().__init__(name, rng)

        self.sklearn_model = GradientBoostingClassifier(random_state=rng, n_estimators=n_estimators)


