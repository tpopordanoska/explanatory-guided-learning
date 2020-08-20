import abc

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.results.per_strategy import ResultsPerStrategy
from src.utils.normalizer import Normalizer
from src.utils.plotting import plot_decision_surface


class RunningInstance:

    def __init__(self, args, experiment, known_idx, train_idx, test_idx, param):
        self.args = args
        self.experiment = experiment
        self.known_idx = known_idx
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.param = param

        self.results = ResultsPerStrategy()
        self.iteration = 0
        self.annotated_point = {}
        self.cos_distance_matrix = None

    @abc.abstractmethod
    def query(self):
        pass

    def get_all_data(self):
        X_known, X_train = self.get_known_train_features()
        y_known, y_train =  self.experiment.y[self.known_idx],  self.experiment.y[self.train_idx]
        X_test, y_test = self.get_from_indexes(self.experiment.X, self.test_idx), self.experiment.y[self.test_idx]

        return X_known, y_known, X_train, y_train, X_test, y_test

    def get_known_train_features(self):
        X_known = self.get_from_indexes(self.experiment.X, self.known_idx)
        X_train = self.get_from_indexes(self.experiment.X, self.train_idx)

        return X_known, X_train

    def get_margins(self, data):
        if hasattr(self.experiment.model.sklearn_model, "decision_function"):
            margins = np.abs(self.experiment.model.decision_function(data))
        elif hasattr(self.experiment.model.sklearn_model, "predict_proba"):
            probs = self.experiment.model.predict_proba(data)
            margins = np.sum(probs * np.log(probs), axis=1).ravel()
        else:
            raise AttributeError("Model with either decision_function or predict_proba method")

        return margins

    def calc_acc(self, y_true, y_pred, metric="f1"):
        """
        Calculate the accuracy.

        :param y_true: The true labels
        :param y_pred: The predictions
        :param metric: The metric to use

        :return: The score
        """
        if metric == "auc":
            return self.get_auc_score(y_true, y_pred)
        return self.get_f1_score(y_true, y_pred)

    def get_f1_score(self, y_true, y_pred):
        score = f1_score(y_true, y_pred, average='macro')
        self.experiment.file.write("F1 score: {}\n".format(score))
        return score

    def get_auc_score(self, y_true, y_pred):
        try:
            score = roc_auc_score(y_true, y_pred)
        except ValueError:
            score = 0
        self.experiment.file.write("AUC score: {}\n".format(score))
        return score

    def train(self):
        """
        Train the model and calculate the accuracy.
        """
        X_known, y_known, X_train, y_train, X_test, y_test = self.get_all_data()
        X_known, X_train, X_test = Normalizer(self.experiment.normalizer).normalize_all(X_known, X_train, X_test)

        self.experiment.model.fit(X_known, y_known)
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        self.results.scores_f1.append(self.calc_acc(y_train, y_pred_train, "f1"))
        self.results.test_scores_f1.append(self.calc_acc(y_test, y_pred_test, "f1"))
        self.results.scores_auc.append(self.calc_acc(y_train, y_pred_train, "auc"))
        self.results.test_scores_auc.append(self.calc_acc(y_test, y_pred_test, "auc"))

    def predict(self, X):
        return self.experiment.model.predict(X)

    def normalize_and_predict(self, X):
        X_known, _ = self.get_known_train_features()
        _, X_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X)

        return self.predict(X_norm)

    def label_query(self, query_idx):
        """
        Query an "oracle": move the selected query from the train to the known dataset.

        :param query_idx: The index of the query to be moved

        :return: The new indexes of the known and train set
        """
        known_idx, train_idx = set(self.known_idx), set(self.train_idx)
        assert query_idx in self.train_idx and not query_idx in self.known_idx
        self.known_idx = np.array(list(sorted(known_idx | {query_idx})))
        self.train_idx = np.array(list(sorted(train_idx - {query_idx})))

    def calculate_query_accuracy(self, query_idx):
        """
        Calculate score of the query for measuring narrative bias

        :param query_idx: The index of the selected query in X

        """
        X_known, X_train = self.get_known_train_features()
        _, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)

        if query_idx:
            idx_in_train = np.where(self.train_idx == query_idx)[0][0]
            query_predicted = self.predict(self.get_from_indexes(X_train_norm, idx_in_train).reshape(1, -1))
            self.results.query_scores.append(int(self.experiment.y[query_idx] == query_predicted[0]))

    def plot(self, title, query_idx=None, y_pred=None):
        if self.args.plots_on:
            plot_decision_surface(self,
                                  query_idx=query_idx,
                                  y_pred=y_pred,
                                  title=title)

    @staticmethod
    def get_from_indexes(X, indexes):
        if isinstance(X, pd.DataFrame):
            return X.iloc[indexes]
        return X[indexes]

    @staticmethod
    def select_random(data, rng):
        """
        Get a random element from the given data.

        :param data: The data to find a random element from
        :param rng: RandomState object

        :return: A random element from the data
        """
        return rng.choice(data)
