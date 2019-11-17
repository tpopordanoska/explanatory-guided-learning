import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from . import select_random, select_by_coordinates, least_confident_idx, plot_decision_surface


class ActiveLearningLoop:

    def run(self, model, X_train, y_train, X_unlabeled, y_unlabeled, max_iter=1, method='al_least_confident'):

        model.fit(X_train, y_train)
        acc_scores = []

        for iteration in range(max_iter):
            # Find the instance to ask label for
            if method == "al_least_confident":
                query_idx = least_confident_idx(model, X_unlabeled)
            elif method == "sq_random":
                # Find a random point from the grid points to label it
                remaining_grid_points = X_unlabeled[np.where(y_unlabeled == 1)]
                selected_point = select_random(remaining_grid_points)
                query_idx = np.where((X_unlabeled[:, 0] == selected_point[0]) &
                                      (X_unlabeled[:, 1] == selected_point[1]))[0][0]
            elif method == "sq_coordinates":
                # TODO: allow user to enter the coordinates of the point
                query_idx = select_by_coordinates(4, 3, X_unlabeled)
            print("Selected point: ", X_unlabeled[query_idx])
            plot_decision_surface(model,
                                  X_train,
                                  y_train,
                                  X_unlabeled,
                                  y_unlabeled,
                                  least_conf=X_unlabeled[query_idx],
                                  title="Model: " + model.model_name)
            print("Labeled data before selection: ", X_train.shape, y_train.shape)
            print("Unlabeled data before selection: ", X_unlabeled.shape, y_unlabeled.shape)

            # Query an "oracle" and add the labeled example to the training set
            X_train = np.concatenate((X_train, [X_unlabeled[query_idx]]), axis=0)
            y_train = np.concatenate((y_train, [y_unlabeled[query_idx]]), axis=0)

            # Remove the newly labeled instance from the "unlabeled" set
            X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)
            print("Labeled data after selection: ", X_train.shape, y_train.shape)
            print("Unlabeled data after selection: ", X_unlabeled.shape, y_unlabeled.shape)

            # Calculate the accuracy of the model
            y_pred = model.predict(X_unlabeled)
            score = accuracy_score(y_unlabeled, y_pred)
            acc_scores.append(score)
            print("Accuracy score:", score)
            print(classification_report(y_unlabeled, y_pred))
            # plot_decision_surface(model, X_train, y_train, X_unlabeled, y_unlabeled, X_unlabeled[query_idx])

            # Retrain the model with the new traning set
            model.fit(X_train, y_train)

        plot_decision_surface(model,
                              X_train,
                              y_train,
                              X_unlabeled,
                              y_unlabeled,
                              least_conf=X_unlabeled[query_idx],
                              title="Model: " + model.model_name)
        return X_train, y_train, X_unlabeled, y_unlabeled, y_pred, acc_scores