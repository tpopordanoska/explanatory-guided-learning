from sklearn.metrics import f1_score, classification_report

from src import *
from src import plot_decision_surface
from src.clustering import run_kmedoids
from src.optimal_user import Annotator
import numpy as np


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

    def run(self, model, experiment, known_idx, train_idx, test_idx, max_iter=1, method='al_least_confident', path=None):

        X_known, y_known = experiment.X[known_idx], experiment.y[known_idx]
        X_train, y_train = experiment.X[train_idx], experiment.y[train_idx]
        X_test, y_test = experiment.X[test_idx], experiment.y[test_idx]
        rng = experiment.rng

        plot_points(X_known, y_known, "The known points", path)
        plot_points(X_test, y_test, "The test points", path)

        model.fit(X_known, y_known)

        acc_scores = []
        y_pred = model.predict(X_train)
        score = f1_score(y_train, y_pred)
        acc_scores.append(score)
        print("F1 score:", score)
        print(classification_report(y_train, y_pred))

        test_acc_scores = []
        y_pred_test = model.predict(X_test)
        test_score = f1_score(y_test, y_pred_test)
        test_acc_scores.append(test_score)
        print("F1 score:", test_score)
        print(classification_report(y_test, y_pred_test))

        for iteration in range(max_iter):
            # If we have selected all instances in the train dataset
            if not len(train_idx):
                break

            # Find the instance to ask label for
            if method == "al_least_confident":
                # Find a point closest to the decision boundary
                query_idx = least_confident_idx(model, X_train)
            elif method == "sq_random":
                # Class conditional random sampling
                elements, counts = np.unique(y_known, return_counts=True)
                total = sum(counts)
                weights = [counts[1] / total, counts[0] / total]
                sampled_class = rng.choice(elements, p=weights)
                sampled_subset = X_train[np.where(y_train == sampled_class)]
                # If the pool of the selected class is exhausted, switch to random sampling from the other class
                if not len(sampled_subset):
                    sampled_subset = X_train[np.where(y_train != sampled_class)]
                selected_point = select_random(sampled_subset)
                query_idx = np.where((X_train[:, 0] == selected_point[0]) &
                                     (X_train[:, 1] == selected_point[1]))[0][0]
            elif method == "optimal_user":
                # Find the wrong point closest to the centroid
                points_unlabeled = np.concatenate((X_train, np.array([y_train]).T, np.array([y_pred]).T), axis=1)
                points_all = concatenate_data(X_known, y_known, X_train, y_train, y_pred)
                clusters, centroids = run_kmedoids(points_all, 10, rng=rng, path=path, method=method)
                wrong_points, query_idx, query = Annotator(points_unlabeled).select_closest(points_all, clusters)
                if not len(wrong_points):
                    break
                query_idx = np.where((X_train[:, 0] == query[0]) & (X_train[:, 1] == query[1]))[0][0]
                run_kmedoids(points_all, 10, other_points=wrong_points, rng=rng, path=path, method=method)

            print("Selected point: ", X_train[query_idx])
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

            # known_idx, train_idx = self.move(known_idx, train_idx, query_idx)
            # X_known, y_known = experiment.X[known_idx], experiment.y[known_idx]
            # X_train, y_train = experiment.X[train_idx], experiment.y[train_idx]

            print("Known data after selection: ", X_known.shape, y_known.shape)
            print("Train data after selection: ", X_train.shape, y_train.shape)


            y_pred_test = model.predict(X_test)
            test_score = f1_score(y_test, y_pred_test)
            test_acc_scores.append(test_score)
            print("F1 score:", test_score)
            print(classification_report(y_test, y_pred_test))

            # Retrain the model with the new traning set
            model.fit(X_known, y_known)

            # Calculate the accuracy of the model
            y_pred = model.predict(X_train)
            score = f1_score(y_train, y_pred)
            acc_scores.append(score)
            print("F1 score:", score)
            print(classification_report(y_train, y_pred))
            # plot_decision_surface(model, X_train, y_train, X_unlabeled, y_unlabeled, X_unlabeled[query_idx])

        if query_idx is not None:
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