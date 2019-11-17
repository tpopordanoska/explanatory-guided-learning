from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


class Learner:
    """
    Class containing multiple sklearn models
    """

    def __init__(self, model_name=""):
        self._model = None
        self.model_name = model_name

    def fit(self, X, y):
        self._model.fit(X, y)

    def decision_function(self, X):
        return self._model.decision_function(X)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class SVM(Learner):
    def __init__(self, model_name='svm'):
        super().__init__(model_name)

        model = None
        if model_name == 'svm':
            model = LinearSVC(
                penalty='l2',
                loss='hinge',
                multi_class='ovr',
                random_state=0)

        elif model_name == 'svm_rbf':
            model = SVC(gamma=1, C=1e2)

        elif model_name == 'l1svm':
            model = LinearSVC(
                penalty='l1',
                loss='squared_hinge',
                dual=False,
                multi_class='ovr',
                random_state=0)

        self._model = model


class LogRegression(Learner):
    def __init__(self, model_name='Logistic Regression'):
        super().__init__(model_name)

        self._model = LogisticRegression(
            penalty='l2',
            multi_class='ovr',
            fit_intercept=False,
            random_state=0)


class GradientBoosting(Learner):
    def __init__(self, model_name="Gradient Boosting"):
        super().__init__(model_name)

        kwargs = {
            'n_estimators': 1200,
            'max_depth': 3,
            'subsample': 0.5,
            'learning_rate': 0.01,
            'min_samples_leaf': 1,
            'random_state': 3,
        }
        self._model = GradientBoostingClassifier(**kwargs)

