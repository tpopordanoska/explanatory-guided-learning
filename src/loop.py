from sklearn.metrics import f1_score, classification_report, roc_curve, auc

from src import *
from src import plot_decision_surface
from src.clustering import run_kmedoids
from src.experiments import *
from src.optimal_user import Annotator
import numpy as np
import pandas as pd

class ActiveLearningLoop:

    @staticmethod
    def move(known_idx, train_idx, query_idx):
        known_idx, train_idx = set(known_idx), set(train_idx)
        # assert query_idx in train_idx and not query_idx in known_idx
        # Query an "oracle" and add the labeled example to the training set
        known_idx = np.array(list(sorted(known_idx | {query_idx})))
        # Remove the newly labeled instance from the "unlabeled" set
        train_idx = np.array(list(sorted(train_idx - {query_idx})))
        return known_idx, train_idx

    @staticmethod
    def get_print_f1_score(y_true, y_pred):
        score = f1_score(y_true, y_pred)
        print("F1 score:", score)
        print(classification_report(y_true, y_pred))
        return score

    @staticmethod
    def get_auc_score( y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
        return auc(fpr, tpr)

    def calc_acc(self, y_train, y_pred, y_test, y_pred_test, acc_scores, test_acc_scores, metric="f1"):
        if metric == "auc":
            score = self.get_auc_score(y_train, y_pred)
            test_score = self.get_auc_score(y_test, y_pred_test)
        else:
            score = self.get_print_f1_score(y_train, y_pred)
            test_score = self.get_print_f1_score(y_test, y_pred_test)
        acc_scores.append(score)
        test_acc_scores.append(test_score)

    def search_query(self, pd_train, y_known, rng):
        # Class conditional random sampling
        elements, counts = np.unique(y_known, return_counts=True)
        total = sum(counts)
        weights = [counts[1] / total, counts[0] / total]
        sampled_class = rng.choice(elements, p=weights)
        pd_sampled_subset = pd_train[pd_train.labels == sampled_class]
        # If the pool of the selected class is exhausted, switch to random sampling from the other class
        if pd_sampled_subset.empty:
            pd_sampled_subset = pd_train[pd_train.labels != sampled_class]
        selected_point = select_random(pd_sampled_subset, rng)
        # selected_point.name returns the index of the point in pd_train
        return selected_point.name

    def optimal_user(self, points_all, pd_all_points, pd_train, how_many_clusters, rng, path, method, theta):
        # Drop the last column (true labels) for kmedoids
        clusters, centroids = run_kmedoids(points_all[:, :-1], how_many_clusters, rng=rng, path=path,
                                           method=method)
        wrong_points, query_idx_in_pd_all_points, query = Annotator(pd_train)\
            .select_from_worst_cluster(pd_all_points, clusters, theta=theta, rng=rng)
        print("Number of wrong points: ", len(wrong_points))
        if not len(wrong_points):
            return None
        run_kmedoids(points_all[:, :-1], how_many_clusters, other_points=wrong_points, rng=rng, path=path,
                     method=method)
        pd_train_features = pd_train.drop(columns=["labels", "predictions"])
        query_features = query.drop(columns=["labels", "predictions"])
        return pd_train_features[pd_train_features == query_features].dropna().index[0]

    def run(self, model, experiment, known_idx, train_idx, test_idx, max_iter=1, method='al_least_confident', path=None, how_many_clusters=10):

        acc_scores = []
        test_acc_scores = []

        if isinstance(experiment.X, pd.DataFrame):
            X_known, y_known = experiment.X.iloc[known_idx].to_numpy(), experiment.y[known_idx]
            X_train, y_train = experiment.X.iloc[train_idx].to_numpy(), experiment.y[train_idx]
            X_test, y_test = experiment.X.iloc[test_idx].to_numpy(), experiment.y[test_idx]
        else:
            X_known, y_known = experiment.X[known_idx], experiment.y[known_idx]
            X_train, y_train = experiment.X[train_idx], experiment.y[train_idx]
            X_test, y_test = experiment.X[test_idx], experiment.y[test_idx]

        if isinstance(experiment, Synthetic):
            plot_points(X_known, y_known, "The known points", path)
            plot_points(X_test, y_test, "The test points", path)

        model.fit(X_known, y_known)
        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        self.calc_acc(y_train, y_pred, y_test, y_pred_test, acc_scores, test_acc_scores, metric=experiment.metric)


        feature_names = experiment.feature_names.copy()
        feature_names.extend(['predictions', 'labels'])

        for iteration in range(max_iter):
            # If we have selected all instances in the train dataset
            if not len(X_train):
                break

            points_train = np.concatenate((X_train, np.array([y_train]).T, np.array([y_pred]).T), axis=1)
            points_all = concatenate_data(X_known, y_known, X_train, y_train, y_pred)

            pd_all_points = pd.DataFrame(data=points_all, columns=feature_names)
            pd_train = pd.DataFrame(data=points_train, columns=feature_names)

            # Find the instance to ask label for
            if method == "al_least_confident":
                # Find a point closest to the decision boundary
                query_idx = least_confident_idx(model, X_train)
            elif method == "sq_random":
                # Class conditional random sampling
                query_idx = self.search_query(pd_train, y_known, experiment.rng)
            elif method == "optimal_user_t1":
                # Find the wrong point closest to the centroid of the worst cluster
                query_idx = self.optimal_user(points_all, pd_all_points, pd_train, how_many_clusters, experiment.rng, path, method, 0.1)
            elif method == "optimal_user_t2":
                # Find the wrong point closest to the centroid of the worst cluster
                query_idx = self.optimal_user(points_all, pd_all_points, pd_train, how_many_clusters, experiment.rng, path, method, 1)
            elif method == "optimal_user_t3":
                # Find the wrong point closest to the centroid of the worst cluster
                query_idx = self.optimal_user(points_all, pd_all_points, pd_train, how_many_clusters, experiment.rng, path, method, 10)

            if query_idx is None:
                break

            print("Selected point: ", X_train[query_idx])
            if isinstance(experiment, Synthetic):
                plot_decision_surface(model,
                                      X_known,
                                      y_known,
                                      X_train,
                                      y_train,
                                      least_conf=X_train[query_idx],
                                      title=model.model_name + " " + method,
                                      path=path)

            # Move the selected query from the train to the known dataset
            print("Known data before selection: ", X_known.shape, y_known.shape)
            print("Train data before selection: ", X_train.shape, y_train.shape)
            # Query an "oracle" and add the labeled example to the training set
            X_known = np.concatenate((X_known, [X_train[query_idx]]), axis=0)
            y_known = np.concatenate((y_known, [y_train[query_idx]]), axis=0)
            # Remove the newly labeled instance from the "unlabeled" set
            X_train = np.delete(X_train, query_idx, axis=0)
            y_train = np.delete(y_train, query_idx, axis=0)
            print("Known data after selection: ", X_known.shape, y_known.shape)
            print("Train data after selection: ", X_train.shape, y_train.shape)

            if not len(X_train):
                break
            # Retrain the model with the new training set
            model.fit(X_known, y_known)

            y_pred = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            self.calc_acc(y_train, y_pred, y_test, y_pred_test, acc_scores, test_acc_scores, metric=experiment.metric)
            # plot_decision_surface(model, X_train, y_train, X_unlabeled, y_unlabeled, X_unlabeled[query_idx])

        if isinstance(experiment, Synthetic):
            if query_idx is not None and len(X_train):
                plot_decision_surface(model,
                                      X_known,
                                      y_known,
                                      X_train,
                                      y_train,
                                      least_conf=X_train[query_idx],
                                      title=model.model_name + " " + method,
                                      path=path)

            # Plot the predictions
            plot_decision_surface(model,
                                  X_known,
                                  y_known,
                                  X_test,
                                  y_test,
                                  y_pred=y_pred_test,
                                  title="Predictions of the model on test data" + model.model_name + " method " + method,
                                  path=path)

        return acc_scores, test_acc_scores