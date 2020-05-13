from sklearn.metrics import f1_score, roc_auc_score
from skrules import SkopeRules

from src.clustering import run_kmedoids
from src.plotting import *
from src.xgl import Annotator


class LearningLoop:

    def __init__(self, experiment, no_clusters, max_iter, path, file, plots_off, use_weights, use_labels):
        self.experiment = experiment
        self.no_clusters = no_clusters
        self.max_iter = max_iter
        self.path = path
        self.file = file
        self.plots_off = plots_off
        self.use_weights = use_weights
        self.use_labels = use_labels
        self.query_array = []
        self.annotated_point = {}
        self.scores_f1 = []
        self.test_scores_f1 = []
        self.scores_auc = []
        self.test_scores_auc = []
        self.query_scores = []

        self.METHODS = {
            "random": lambda  **kwargs: self.random_sampling(**kwargs),
            "al_least_confident": lambda **kwargs: self.least_confident_idx(**kwargs),
            "sq_random": lambda **kwargs: self.search_query_array(**kwargs),
            "xgl": lambda **kwargs: self.xgl_clustering(**kwargs),
            "rules": lambda **kwargs: self.xgl_rules(**kwargs),
            "rules_hierarchy": lambda **kwargs: self.xgl_rules_hierarchy(**kwargs)
        }

    def random_sampling(self, **kwargs):
        """
        Get the index of a random point to be labeled.

        :param kwargs: Keyword arguments

        :return: The index of the query (in X) to be labeled
        """
        train_idx = kwargs.pop("train_idx")
        return select_random(train_idx, self.experiment.model.rng)

    def least_confident_idx(self, **kwargs):
        """
        Get the index of the example closest to the decision boundary.

        :param kwargs: Keyword arguments

        :return: The index (in X) of the least confident example
        """
        experiment = self.experiment
        known_idx = kwargs.pop("known_idx")
        train_idx = kwargs.pop("train_idx")
        X_known = get_from_indexes(experiment.X, known_idx)
        X_train = get_from_indexes(experiment.X, train_idx)

        _, X_train_norm = Normalizer(experiment.normalizer).normalize_known_train(X_known, X_train)

        if hasattr(experiment.model.sklearn_model, "decision_function"):
            margins = np.abs(experiment.model.decision_function(X_train_norm))
        elif hasattr(experiment.model.sklearn_model, "predict_proba"):
            probs = experiment.model.predict_proba(X_train_norm)
            margins = np.sum(probs * np.log(probs), axis=1).ravel()
        else:
            raise AttributeError("Model with either decision_function or predict_proba method")

        if len(margins) == 0:
            return None
        return train_idx[np.argmin(margins)]

    def search_query_array(self, **kwargs):
        """
        Find index of the query to be labeled using the search_query strategy.

        :param kwargs: Keyword arguments

        :return: The index of the query (in X) to be labeled
        """
        train_idx = kwargs.pop("train_idx")
        iteration = kwargs.pop("iteration")
        y_train = self.experiment.y[train_idx]

        if not len(self.query_array):
            # It has not been initialized
            queried = 0
            query_array = []
            # while we haven't emptied the training pool and we haven't reached the max_iter
            while len(y_train) and queried < self.max_iter:
                for label in np.unique(y_train):
                    sampled_subset_idx = np.where(y_train == label)[0]
                    if not len(sampled_subset_idx):
                        continue
                    idx_in_train = select_random(sampled_subset_idx, self.experiment.rng)
                    query_array.append(train_idx[idx_in_train])
                    train_idx = np.delete(train_idx, idx_in_train)
                    queried += 1
                    y_train = self.experiment.y[train_idx]
            self.query_array = query_array

        return self.query_array[iteration]

    def xgl_clustering(self, **kwargs):
        """
        Find index of the query to be labeled using the XGL strategy.

        :param kwargs: Keyword arguments

        :return: The index of the query (in X) to be labeled
        """
        train_idx = kwargs.pop("train_idx")
        known_idx = kwargs.pop("known_idx")
        y_pred = kwargs.pop("y_pred")
        theta = kwargs.pop("theta")
        iteration = kwargs.pop("iteration")
        X_train = get_from_indexes(self.experiment.X, train_idx)
        X_known = get_from_indexes(self.experiment.X, known_idx)

        X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)

        X_known_train = np.concatenate((X_known_norm, X_train_norm), axis=0)
        kmedoids_pd = pd.DataFrame(data=X_known_train)
        kmedoids_pd['predictions'] = np.concatenate((self.experiment.y[known_idx], y_pred), axis=0)

        known_train_pd = kmedoids_pd.copy()
        known_train_idx = np.concatenate((known_idx, train_idx), axis=0)
        known_train_pd['labels'] = self.experiment.y[known_train_idx]
        known_train_pd["idx"] = known_train_idx

        # Find the clusters and their centroids
        clusters, centroids = run_kmedoids(kmedoids_pd, n_clusters=self.no_clusters, use_labels=self.use_labels,
                                           use_weights=self.use_weights, path=self.path, plots_off=self.plots_off)
        # Find the index of the query to be labeled
        wrong_points, query_idx = Annotator().select_from_worst_cluster(known_train_pd, clusters, train_idx, theta=theta,
                                                                        rng=self.experiment.rng, file=self.file)
        key = "xgl_" + str(theta)
        if not len(wrong_points) and key not in self.annotated_point.keys():
            self.annotated_point[key] = iteration

        # Plot the wrong points
        if len(wrong_points):
            run_kmedoids(kmedoids_pd, n_clusters=self.no_clusters, other_points=wrong_points, use_labels=self.use_labels,
                         use_weights=self.use_weights, path=self.path, plots_off=self.plots_off)

        return query_idx

    def xgl_rules(self, hierarchy=False, **kwargs):
        """
        Find index of the query to be labeled with the XGL strategy using global surrogate model (decision trees).

        :param hierarchy: Whether to do hierarchical global explanation
        :param kwargs: Keyword arguments

        :return: The index of the query (in X) to be labeled
        """
        train_idx = kwargs.get("train_idx")
        known_idx = kwargs.get("known_idx")
        theta = kwargs.get("theta")

        X_train = get_from_indexes(self.experiment.X, train_idx)
        X_known = get_from_indexes(self.experiment.X, known_idx)
        X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)

        X_known_train = np.concatenate((X_known_norm, X_train_norm), axis=0)
        y_predicted = self.experiment.model.predict(X_known_train)

        # Create a dataframe holding the known and train data with their predictions, labels and indexes in X
        X_known_train_pd = pd.DataFrame(data=X_known_train, columns=self.experiment.feature_names)
        X_known_train_pd['is_train'] = np.concatenate([[False] * len(known_idx), [True] * len(train_idx)])
        X_known_train_pd['predictions'] = y_predicted
        X_known_train_pd['labels'] = np.concatenate([self.experiment.y[known_idx], self.experiment.y[train_idx]])
        X_known_train_pd["idx_in_X"] = np.concatenate([known_idx, train_idx])
        X_train_pd = X_known_train_pd[X_known_train_pd["is_train"]]

        X_known_train_features = X_known_train_pd.drop(['is_train', 'predictions', 'labels', 'idx_in_X'], axis=1)

        column_names = self.experiment.feature_names
        if X_known_train_features.shape[1] != len(column_names):
            column_names = ['Col_' + str(i) for i in range(0, X_known_train_features.shape[1])]

        n_estimators = [5, 10, 20, 35, 50]
        num_features = len(self.experiment.feature_names)
        max_depth = num_features * 2 if num_features < 5 else num_features
        for n_estim in n_estimators:
            clf = SkopeRules(n_estimators=n_estim,
                             precision_min=0.4,
                             recall_min=0.01,
                             max_depth=max_depth,
                             max_features=None,
                             max_samples=1.0,
                             max_depth_duplication=n_estim,
                             random_state=self.experiment.rng,
                             feature_names=column_names)

            clf.fit(X_known_train_features, X_known_train_pd['predictions'])
            if self.compare_performance(clf, 0.2, **kwargs):
                break

        # Get the worst rule
        worst_rule = self.get_worst_rule(X_known_train_pd, theta, clf)

        if hierarchy:
            # Find the points that satisfy the chosen rule
            points_known_train_pd = self.get_points_satisfying_rule(X_known_train_pd, worst_rule[0])
            # Check if all predictions are the same
            kt_predictions = points_known_train_pd.predictions.to_numpy()
            if not all(element == kt_predictions[0] for element in kt_predictions):
                # Get worst rule from the new points
                print("Hierarchical rule selection")
                worst_rule = self.get_worst_rule(points_known_train_pd, theta, clf)
                X_train_pd = points_known_train_pd[points_known_train_pd["is_train"]]

        # Find the points that satisfy the chosen rule
        points_pd = self.get_points_satisfying_rule(X_train_pd, worst_rule[0])
        # From those, find the ones that are wrongly classified wrt the rules
        points_predictions = np.ones(len(points_pd)) * worst_rule[1]
        wrong_points_idx = points_pd[points_pd.labels != points_predictions]["idx_in_X"]

        if len(wrong_points_idx) == 0:
            return select_random(train_idx, self.experiment.rng)

        return select_random(wrong_points_idx, self.experiment.rng)

    def get_data(self, **kwargs):
        experiment = self.experiment
        train_idx = kwargs.pop("train_idx")
        known_idx = kwargs.pop("known_idx")
        test_idx = kwargs.pop("test_idx")

        X_known, y_known = get_from_indexes(experiment.X, known_idx), experiment.y[known_idx]
        X_train, y_train = get_from_indexes(experiment.X, train_idx), experiment.y[train_idx]
        X_test, y_test = get_from_indexes(experiment.X, test_idx), experiment.y[test_idx]

        return X_known, y_known, X_train, y_train, X_test, y_test

    def compare_performance(self, clf, threshold, **kwargs):
        X_known, y_known, X_train, y_train, X_test, y_test = self.get_data(**kwargs)
        # Normalize the data
        X_known, X_train, X_test = Normalizer(self.experiment.normalizer).normalize_all(X_known, X_train, X_test)

        # Calculate accuracy wrt the extracted rules
        y_pred_test_rules = clf.predict(X_test)
        score_rules = self.calc_acc(y_test, y_pred_test_rules, "f1")

        # Get accuracy of the blackbox classifier
        iteration = kwargs.get("iteration")
        score_blackbox = self.scores_f1[iteration]

        return np.abs(score_blackbox - score_rules) < threshold

    def get_worst_rule(self, data, theta, clf):
        X_known_train_features = data.drop(['is_train', 'predictions', 'labels', 'idx_in_X'], axis=1)
        predictions = data['predictions']
        X_train_pd = data[data["is_train"]]

        rules = []
        for idx, features in enumerate(np.unique(self.experiment.y)):
            clf.fit(X_known_train_features, predictions == idx)
            for rule in clf.rules_:
                # Each element in rules_ is a tuple (rule, precision, recall, nb).
                # nb = the number of time that this rule was extracted from the trees built during skope-rules' fitting
                # Get the points in X_train
                points_pd = self.get_points_satisfying_rule(X_train_pd, rule[0])
                # if there are no points in X_train that satisfy the rule, skip it
                if len(points_pd) > 0:
                    points_predictions = np.ones(len(points_pd)) * idx
                    score = self.get_f1_score(points_pd.labels, points_predictions)
                    score_blackbox = self.get_f1_score(points_pd.labels, points_pd.predictions)
                    # Each element in rules is a tuple (rule, class, f1_score wrt rules, f1_score wrt classifier)
                    rules.append((rule[0], idx, score, score_blackbox))
            if not self.plots_off:
                plot_rules(clf, data, predictions == idx, idx, self.path)

        self.file.write("Parameters for Skope Rules: {}".format(clf.get_params()))
        print("Number of extracted rules: ", len(rules))

        # Sort by the f1_score wrt the rules
        rules.sort(key=lambda x: x[2], reverse=True)

        logits = [x[2] for x in rules]
        exps = [np.exp(i * theta - max(logits)) for i in logits]
        softmax = [j / sum(exps) for j in exps]
        selected_rule_idx = self.experiment.rng.choice(len(rules), p=softmax)

        return rules[selected_rule_idx]

    def xgl_rules_hierarchy(self, **kwargs):
        """
        Find index of the query to be labeled with the XGL strategy using global surrogate model (decision trees).

        :param kwargs: Keyword arguments

        :return: The index of the query (in X) to be labeled
        """
        return self.xgl_rules(**kwargs, hierarchy=True)

    def get_points_satisfying_rule(self, X, rule):
        return X.query(rule)

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

    def train_and_get_acc(self, known_idx, train_idx, test_idx, query_idx=None):
        """
        Train the model and calculate the accuracy.

        :param known_idx: The indexes of the known set
        :param train_idx: The indexes of the train set
        :param test_idx: The indexes of the test set
        :param query_idx: The index of the selected query in X

        :return: The predictions and the lists with the scores
        """
        experiment = self.experiment
        X_known, y_known = get_from_indexes(experiment.X, known_idx), experiment.y[known_idx]
        X_train, y_train = get_from_indexes(experiment.X, train_idx), experiment.y[train_idx]
        X_test, y_test = get_from_indexes(experiment.X, test_idx), experiment.y[test_idx]

        # Normalize the data
        X_known, X_train, X_test = Normalizer(self.experiment.normalizer).normalize_all(X_known, X_train, X_test)

        # Calculate score of the query for measuring narrative bias
        if query_idx:
            idx_in_known = np.where(known_idx == query_idx)[0][0]
            query_predicted = experiment.model.predict(get_from_indexes(X_known, idx_in_known).reshape(1, -1))
            self.query_scores.append(int(experiment.y[query_idx] == query_predicted[0]))

        experiment.model.fit(X_known, y_known)
        y_pred = experiment.model.predict(X_train)
        y_pred_test = experiment.model.predict(X_test)

        self.scores_f1.append(self.calc_acc(y_train, y_pred, "f1"))
        self.test_scores_f1.append(self.calc_acc(y_test, y_pred_test, "f1"))
        self.scores_auc.append(self.calc_acc(y_train, y_pred, "auc"))
        self.test_scores_auc.append(self.calc_acc(y_test, y_pred_test, "auc"))

        return y_pred, y_pred_test

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

        score = f1_score(y_true, y_pred, average='macro')
        if self.file is not None:
            self.file.write("F1 score: {}\n".format(score))
        else:
            print("F1 score: {}\n".format(score))
        return score

    def get_auc_score(self, y_true, y_pred):
        """
        Calculate AUC score.

        :param y_true: The true labels
        :param y_pred: The predictions

        :return: The AUC score
        """
        try:
            score = roc_auc_score(y_true, y_pred)
        except ValueError:
            score = 0
            self.file.write("Setting AUC score to 0 because of ValueError.")
        if self.file is not None:
            self.file.write("AUC score: {}\n".format(score))
        else:
            print("AUC score: {}\n".format(score))
        return score

    def run(self, method, known_idx, train_idx, test_idx):
        """
        Perform the learning loop: train a model, select a query and retrain the model until the budget is exhausted.

        :param method: The method to be used for query selection
        :param known_idx: The indexes of the known set
        :param train_idx: The indexes of the train set
        :param test_idx: The indexes of the test set

        :return: List of accuracy scores (F1 and AUC) on train and test set
        """
        self.query_array = []
        self.scores_f1 = []
        self.test_scores_f1 = []
        self.scores_auc = []
        self.test_scores_auc = []
        self.query_scores = []

        experiment = self.experiment
        theta = ""
        if "xgl" in method:
            theta = float(method.split("_")[1])
            method = "xgl"
        elif "rules" in method:
            theta = float(method.split("_")[-1])
            method = "rules"

        # 1. Train a model
        y_pred, y_pred_test = self.train_and_get_acc(known_idx, train_idx, test_idx)

        for iteration in range(self.max_iter):
            self.file.write("Iteration: {}\n".format(iteration))
            # If we have selected all instances in the train dataset
            if len(train_idx) <= 1:
                break

            # 2. Find the index of the query to be labeled
            query_idx = self.METHODS[method](known_idx=known_idx,
                                             train_idx=train_idx,
                                             test_idx=test_idx,
                                             y_pred=y_pred,
                                             iteration=iteration,
                                             theta=theta)

            if query_idx is None:
                break
            self.file.write("Selected point: {}\n".format(get_from_indexes(experiment.X, query_idx)))

            if not self.plots_off:
                plot_decision_surface(experiment,
                                      known_idx,
                                      train_idx,
                                      query_idx=query_idx,
                                      title=str(iteration) + " " + experiment.model.name + " " + method,
                                      path=self.path)

            # 3. Query an "oracle" and add the labeled example to the training set
            known_idx, train_idx = self.move(known_idx, train_idx, query_idx)

            # 4. Retrain the model with the new training set
            y_pred, y_pred_test = self.train_and_get_acc(known_idx, train_idx, test_idx, query_idx)

        # Plot the decision surface
        if not self.plots_off:
            plot_decision_surface(experiment,
                                  known_idx,
                                  train_idx,
                                  title=str(iteration) + " " + experiment.model.name + " " + method,
                                  path=self.path)
            # Plot the predictions
            plot_decision_surface(experiment,
                                  known_idx,
                                  test_idx,
                                  y_pred=y_pred_test,
                                  title="Predictions " + experiment.model.name + " " + method,
                                  path=self.path)

