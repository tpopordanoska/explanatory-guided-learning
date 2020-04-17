from sklearn.metrics import classification_report, f1_score, roc_auc_score
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

        self.METHODS = {
            "random": lambda  **kwargs: self.random_sampling(**kwargs),
            "al_least_confident": lambda **kwargs: self.least_confident_idx(**kwargs),
            "sq_random": lambda **kwargs: self.search_query_array(**kwargs),
            "xgl": lambda **kwargs: self.xgl_clustering(**kwargs),
            "rules": lambda **kwargs: self.xgl_rules(**kwargs),
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

        if hasattr(experiment.model._model, "decision_function"):
            margins = np.abs(experiment.model.decision_function(X_train_norm))
        elif hasattr(experiment.model._model, "predict_proba"):
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

    def xgl_rules(self, **kwargs):
        """
        Find index of the query to be labeled with the XGL strategy using global surrogate model (decision trees).

        :param kwargs: Keyword arguments

        :return: The index of the query (in X) to be labeled
        """
        train_idx = kwargs.pop("train_idx")
        known_idx = kwargs.pop("known_idx")
        y_pred = kwargs.pop("y_pred")

        X_train = get_from_indexes(self.experiment.X, train_idx)
        X_known = get_from_indexes(self.experiment.X, known_idx)
        X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)

        X_known_train = np.concatenate((X_known_norm, X_train_norm), axis=0)
        y_known_predicted = np.concatenate((self.experiment.y[known_idx], y_pred), axis=0)
        X_train_pd = pd.DataFrame(data=X_train_norm)
        X_train_pd['predictions'] = y_pred
        X_train_pd['labels'] = self.experiment.y[train_idx]
        X_train_pd["idx"] = train_idx

        clf = SkopeRules(n_estimators=10,
                         precision_min=0.8,
                         recall_min=0.15,
                         max_depth=5,
                         random_state=self.experiment.rng,
                         feature_names=self.experiment.feature_names)

        self.file.write("Parameters for Skope Rules: {}".format(clf.get_params()))
        clf.fit(X_known_train, y_known_predicted)
        # TODO: Sort by f1:  (2 * x[1][0] * x[1][1]) / (x[1][0] + x[1][1])
        clf.rules_.sort(key=lambda x: x[1][0], reverse=True)
        # Each element in rules_ is a tuple (rule, precision, recall, nb).
        # nb = the number of time that this rule was extracted from the trees built during skope-rules' fitting
        if not clf.rules_:
            return select_random(train_idx, self.experiment.rng)
        print("Best rule: ", clf.rules_[0])
        print("Number of extracted rules: ", len(clf.rules_))
        # self.plot_rules(clf, X_known_train, y_known_predicted)

        if len(clf.rules_) <= self.no_clusters:
            worst_rule = clf.rules_[-1]
        else:
            worst_rule = clf.rules_[self.no_clusters]

        # Find the points that satisfy the chosen rule
        points_idx = self.get_points_satisfying_rule(X_train_norm, worst_rule[0]).index
        points_pd = X_train_pd.iloc[points_idx]
        # From those, find the ones that are wrongly classified
        wrong_points_idx = points_pd[points_pd.labels != points_pd.predictions].idx

        if len(wrong_points_idx) == 0:
            return select_random(train_idx, self.experiment.rng)

        return select_random(wrong_points_idx, self.experiment.rng)

    def plot_rules(self, clf, X, y):

        xx, yy = create_meshgrid(X, 0.05)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, s=45)
        plt.show()

    def get_points_satisfying_rule(self, X, rule):
        X = pd.DataFrame(X, columns=self.experiment.feature_names)
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

    def train_and_get_acc(self, known_idx, train_idx, test_idx, acc_scores_f1, test_acc_scores_f1, acc_scores_auc,
                          test_acc_scores_auc):
        """
        Train the model and calculate the accuracy.

        :param known_idx: The indexes of the known set
        :param train_idx: The indexes of the train set
        :param test_idx: The indexes of the test set
        :param acc_scores_f1: List containing the f1 accuracy scores on train set
        :param test_acc_scores_f1: List containing the f1 accuracy scores on test set
        :param acc_scores_auc: List containing the auc accuracy scores on train set
        :param test_acc_scores_auc: List containing the auc accuracy scores on train set

        :return: The predictions and the lists with the scores
        """
        experiment = self.experiment
        X_known, y_known = get_from_indexes(experiment.X, known_idx), experiment.y[known_idx]
        X_train, y_train = get_from_indexes(experiment.X, train_idx), experiment.y[train_idx]
        X_test, y_test = get_from_indexes(experiment.X, test_idx), experiment.y[test_idx]

        # Normalize the data
        X_known, X_train, X_test = Normalizer(self.experiment.normalizer).normalize_all(X_known, X_train, X_test)

        experiment.model.fit(X_known, y_known)
        y_pred = experiment.model.predict(X_train)
        y_pred_test = experiment.model.predict(X_test)

        acc_scores_f1.append(self.calc_acc(y_train, y_pred, "f1"))
        test_acc_scores_f1.append(self.calc_acc(y_test, y_pred_test, "f1"))
        acc_scores_auc.append(self.calc_acc(y_train, y_pred, "auc"))
        test_acc_scores_auc.append(self.calc_acc(y_test, y_pred_test, "auc"))

        return y_pred, y_pred_test, acc_scores_f1, test_acc_scores_f1, acc_scores_auc, test_acc_scores_auc

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
        experiment = self.experiment
        theta = ""
        if "xgl" in method:
            theta = float(method.split("_")[1])
            method = "xgl"

        # 1. Train a model
        y_pred, y_pred_test, scores_f1, test_scores_f1, scores_auc, test_scores_auc = \
            self.train_and_get_acc(known_idx, train_idx, test_idx, [], [], [], [])

        for iteration in range(self.max_iter):
            self.file.write("Iteration: {}\n".format(iteration))
            # If we have selected all instances in the train dataset
            if len(train_idx) <= 1:
                break

            # 2. Find the index of the query to be labeled
            query_idx = self.METHODS[method](known_idx=known_idx,
                                             train_idx=train_idx,
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
            y_pred, y_pred_test, scores_f1, test_scores_f1, scores_auc, test_scores_auc = self.train_and_get_acc(
                known_idx, train_idx, test_idx, scores_f1, test_scores_f1, scores_auc, test_scores_auc)

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

        return scores_f1, test_scores_f1, scores_auc, test_scores_auc
