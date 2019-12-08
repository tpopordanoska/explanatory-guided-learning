from sklearn.metrics import f1_score, classification_report, roc_curve, auc

from src import *
from src import plot_decision_surface
from src.clustering import run_kmedoids
from src.experiments import *
from src.optimal_user import Annotator
import numpy as np
import pandas as pd

METHODS = {
    "al_least_confident": lambda **kwargs: least_confident_idx(**kwargs),
    "sq_random": lambda **kwargs: search_query(**kwargs),
    "optimal_user_t1": lambda **kwargs: optimal_user(**kwargs, theta=0.1),
    "optimal_user_t2": lambda **kwargs: optimal_user(**kwargs, theta=1),
    "optimal_user_t3": lambda **kwargs: optimal_user(**kwargs, theta=3)
}


def optimal_user(theta, **kwargs):
    """
    Find index of the query to be labeled using the optimal_user strategy.

    :param theta: Parameter for the softmax function
    :param kwargs: Keyword arguments

    :return: The index of the query (in X) to be labeled
    """
    train_idx = kwargs.pop("train_idx")
    known_idx = kwargs.pop("known_idx")
    y_pred = kwargs.pop("y_pred")
    experiment = kwargs.pop("experiment")
    no_clusters = kwargs.pop("no_clusters")
    path = kwargs.pop("path")
    plots_off = kwargs.pop("plots_off")
    rng = experiment.rng

    known_train_idx = np.concatenate((known_idx, train_idx), axis=0)
    kmedoids_pd = pd.DataFrame(experiment.X[known_train_idx], columns=experiment.feature_names)
    kmedoids_pd['predictions'] = np.concatenate((experiment.y[known_idx], y_pred), axis=0)

    known_train_pd = kmedoids_pd.copy()
    known_train_pd['labels'] = experiment.y[known_train_idx]
    known_train_pd["idx"] = known_train_idx

    # Find the clusters and their centroids
    clusters, centroids = run_kmedoids(kmedoids_pd.to_numpy(), no_clusters, path=path, plots_off=plots_off)
    # Find the index of the query to be labeled
    wrong_points, query_idx = Annotator().select_from_worst_cluster(known_train_pd, clusters, theta=theta, rng=rng)
    # Plot the wrong points
    run_kmedoids(kmedoids_pd.to_numpy(), no_clusters, other_points=wrong_points, path=path, plots_off=plots_off)

    return query_idx


def search_query(**kwargs):
    """
    Find index of the query to be labeled using the search_query strategy.

    :param kwargs: Keyword arguments

    :return: The index of the query (in X) to be labeled
    """
    train_idx = kwargs.pop("train_idx")
    experiment = kwargs.pop("experiment")
    X_train, y_train = experiment.X[train_idx], experiment.y[train_idx]

    # Class conditional random sampling
    elements, counts = np.unique(y_train, return_counts=True)
    if len(elements) != len(np.unique(experiment.y)):
        # If the pool of one class is exhausted, switch to random sampling from the other class
        return select_random(train_idx, experiment.rng)

    total = sum(counts)
    weights = [counts[1] / total, counts[0] / total]
    sampled_class = experiment.rng.choice(elements, p=weights)
    sampled_subset_idx = np.where(y_train == sampled_class)[0]
    idx_in_train = select_random(sampled_subset_idx, experiment.rng)
    return train_idx[idx_in_train]


class ActiveLearningLoop:

    def __init__(self, experiment, no_clusters, max_iter, path, file, plots_off):
        self.experiment = experiment
        self.no_clusters = no_clusters
        self.max_iter = max_iter
        self.path = path
        self.file = file
        self.plots_off = plots_off

    def move(self, known_idx, train_idx, query_idx):
        """
        Move the selected query from the train to the known dataset.

        :param known_idx: The indexes of the known set
        :param train_idx: The indexes of the train set
        :param query_idx: The index of the query to be moved

        :return: The new indexes of the known and train set
        """
        self.file.write("Known data before selection: {} {}\n".format(len(known_idx), len(known_idx)))
        self.file.write("Train data before selection: {} {}\n".format(len(train_idx), len(train_idx)))

        known_idx, train_idx = set(known_idx), set(train_idx)
        assert query_idx in train_idx and not query_idx in known_idx
        known_idx = np.array(list(sorted(known_idx | {query_idx})))
        train_idx = np.array(list(sorted(train_idx - {query_idx})))

        self.file.write("Known data after selection: {} {}\n".format(len(known_idx), len(known_idx)))
        self.file.write("Train data after selection: {} {}\n".format(len(train_idx), len(train_idx)))

        return known_idx, train_idx

    def train_and_get_acc(self, known_idx, train_idx, test_idx, acc_scores, test_acc_scores):
        """
        Train the model and calculate the accuracy.

        :param known_idx: The indexes of the known set
        :param train_idx: The indexes of the train set
        :param test_idx: The indexes of the test set
        :param acc_scores: List containing the accuracy scores on train set
        :param test_acc_scores: List containing the accuracy scores on test set

        :return: The predictions and the lists with the scores
        """
        experiment = self.experiment
        X_known, y_known = experiment.X[known_idx], experiment.y[known_idx]
        X_train, y_train = experiment.X[train_idx], experiment.y[train_idx]
        X_test, y_test = experiment.X[test_idx], experiment.y[test_idx]

        experiment.model.fit(X_known, y_known)
        y_pred = experiment.model.predict(X_train)
        y_pred_test = experiment.model.predict(X_test)
        acc_scores.append(self.calc_acc(y_train, y_pred, metric=experiment.metric))
        test_acc_scores.append(self.calc_acc(y_test, y_pred_test, metric=experiment.metric))

        return y_pred, y_pred_test, acc_scores, test_acc_scores

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
        """
        Calculate F1 score.

        :param y_true: The true labels
        :param y_pred: The predictions

        :return: The score
        """

        score = f1_score(y_true, y_pred)
        if self.file is not None:
            self.file.write("F1 score: {}\n".format(score))
            self.file.write(classification_report(y_true, y_pred))
        else:
            print("F1 score: {}\n".format(score))
            print(classification_report(y_true, y_pred))
        return score

    def get_auc_score(self, y_true, y_pred):
        """
        Calculate AUC score.

        :param y_true: The true labels
        :param y_pred: The predictions

        :return: The AUC score
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
        score = auc(fpr, tpr)
        if self.file is not None:
            self.file.write("AUC score: {}\n".format(score))
            self.file.write(classification_report(y_true, y_pred))
        else:
            print("AUC score: {}\n".format(score))
            print(classification_report(y_true, y_pred))
        return score

    def run(self, method, known_idx, train_idx, test_idx):
        """
        Perform the active learning loop: train a model, select a query and retrain the model until budget is exhausted.

        :param method: The method to be used for query selection
        :param known_idx: The indexes of the known set
        :param train_idx: The indexes of the train set
        :param test_idx: The indexes of the test set

        :return: List of accuracy scores on train and test set
        """

        experiment = self.experiment
        if isinstance(experiment, Synthetic):
            plot_points(experiment.X[known_idx], experiment.y[known_idx], "The known points", self.path)
            plot_points(experiment.X[test_idx], experiment.y[test_idx], "The test points", self.path)

        # 1. Train a model
        y_pred, y_pred_test, scores, test_scores = self.train_and_get_acc(known_idx, train_idx, test_idx, [], [])

        for iteration in range(self.max_iter):
            # If we have selected all instances in the train dataset
            if len(train_idx) <= 1:
                break

            # 2. Find the index of the query to be labeled
            query_idx = METHODS[method](experiment=experiment,
                                        known_idx=known_idx,
                                        train_idx=train_idx,
                                        y_pred=y_pred,
                                        no_clusters=self.no_clusters,
                                        path=self.path,
                                        plots_off=self.plots_off)

            if query_idx is None:
                break

            self.file.write("Selected point: {}\n".format(experiment.X[query_idx]))
            if not self.plots_off:
                plot_decision_surface(experiment.model,
                                      experiment.X[known_idx],
                                      experiment.y[known_idx],
                                      experiment.X[train_idx],
                                      experiment.y[train_idx],
                                      least_conf=experiment.X[query_idx],
                                      title="{} {}".format(experiment.model.name, method),
                                      path=self.path)

            # 3. Query an "oracle" and add the labeled example to the training set
            known_idx, train_idx = self.move(known_idx, train_idx, query_idx)

            # 4. Retrain the model with the new training set
            y_pred, y_pred_test, scores, test_scores = self.train_and_get_acc(
                known_idx, train_idx, test_idx, scores, test_scores)

        if not self.plots_off:
            plot_decision_surface(experiment.model,
                                  experiment.X[known_idx],
                                  experiment.y[known_idx],
                                  experiment.X[train_idx],
                                  experiment.y[train_idx],
                                  least_conf=experiment.X[query_idx],
                                  title="{} {}".format(experiment.model.name, method),
                                  path=self.path)
            # Plot the predictions
            plot_decision_surface(experiment.model,
                                  experiment.X[known_idx],
                                  experiment.y[known_idx],
                                  experiment.X[test_idx],
                                  experiment.y[test_idx],
                                  y_pred=y_pred_test,
                                  title="Predicts of the model {} on test data using {}"
                                            .format(experiment.model.name, method),
                                  path=self.path)

        return scores, test_scores
