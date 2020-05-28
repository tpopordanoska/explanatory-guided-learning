import sklearn.metrics.pairwise as metrics
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score
from skrules import SkopeRules
from tqdm import tqdm

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
        self.cos_distance_matrix = None
        self.initial_train_idx = None

        self.METHODS = {
            "random": lambda  **kwargs: self.random_sampling(**kwargs),
            "al_least_confident": lambda **kwargs: self.least_confident_idx(**kwargs),
            "al_density_weighted": lambda **kwargs: self.density_weighted_idx(**kwargs),
            "sq_random": lambda **kwargs: self.search_query_array(**kwargs),
            "xgl": lambda **kwargs: self.xgl_clustering(**kwargs),
            "rules": lambda **kwargs: self.xgl_rules(**kwargs),
            "rules_hierarchy": lambda **kwargs: self.xgl_rules_hierarchy(**kwargs)
        }

    ############## QUERY SELECTION STRATEGIES ##############
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
        train_idx = kwargs.get("train_idx")
        X_known, _, X_train, _, _, _ = self.get_data(**kwargs)

        _, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)
        margins = self.get_margins(X_train_norm)

        if len(margins) == 0:
            return None
        return train_idx[np.argmin(margins)]

    def density_weighted_idx(self, **kwargs):
        """
        Get the index of the next query according to the density-weighted query selection strategy.
        If clustering is used to speed up calculation, similarity of each instance in the train pool to each cluster
        center is calculated.

        :param kwargs: Keyword arguments

        :return: The index (in X) of the selected example to be labeled
        """
        beta = kwargs.pop("param")
        train_idx = kwargs.get("train_idx")
        use_clustering = kwargs.pop("use_clustering")

        X_known, y_known, X_train, y_train, _, _ = self.get_data(**kwargs)
        X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)
        X_known_train = np.concatenate((X_known_norm, X_train_norm), axis=0)
        margins = self.get_margins(X_train_norm)

        if use_clustering:
            kmeans = KMeans().fit(X_known_train)
            predictions_clustering = kmeans.predict(X_train_norm)
            centroids = kmeans.cluster_centers_

            cos_distance = []
            for i in range(len(X_train_norm)):
                # Compare every instance in train to the centroid of the cluster it belongs to
                dist = metrics.cosine_distances(X_train_norm[i].reshape(1, -1),
                                                centroids[predictions_clustering[i]].reshape(1, -1))[0][0]
                cos_distance.append(dist)

        else:
            # Check if distance matrix was previously computed
            dist_matrix_path = os.path.join(os.getcwd(), "distance_matrix")
            try:
                self.cos_distance_matrix = np.load(os.path.join(dist_matrix_path, self.experiment.name + ".npy"))
            except IOError:
                print("File {} does not exist or cannot be read".format(dist_matrix_path))

            # Compute the similarity matrix if it has not been stored already
            if self.cos_distance_matrix is None:
                X_norm = Normalizer(self.experiment.normalizer).normalize(self.experiment.X)
                self.cos_distance_matrix = np.zeros((len(X_norm), len(X_norm)))
                for i, x_i in tqdm(enumerate(X_norm)):
                    for j, x_j in enumerate(X_norm):
                        self.cos_distance_matrix[i][j] = metrics.cosine_distances(
                            x_i.reshape(1, -1), x_j.reshape(1, -1))[0][0]
                try:
                    os.mkdir(dist_matrix_path)
                except FileExistsError:
                    print("Directory {} already exists".format(dist_matrix_path))

                np.save(os.path.join(dist_matrix_path, self.experiment.name), self.cos_distance_matrix)

            cos_distance = []
            for idx in train_idx:
                dists = self.cos_distance_matrix[idx, [train_idx]]
                cos_distance.append(np.mean(dists))

        cos_distance = np.asarray(cos_distance)
        query_idx = np.argmin(margins * cos_distance**beta)

        return train_idx[query_idx]

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
        theta = kwargs.pop("param")
        iteration = kwargs.pop("iteration")
        use_gower = self.experiment.use_gower
        X_train = get_from_indexes(self.experiment.X, train_idx)
        X_known = get_from_indexes(self.experiment.X, known_idx)

        if use_gower:
            # No need to normalize the data, as it is done in the gower_matrix() method
            X_known_train = np.concatenate((X_known, X_train), axis=0)
        else:
            X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)
            X_known_train = np.concatenate((X_known_norm, X_train_norm), axis=0)

        kmedoids_pd = pd.DataFrame(data=X_known_train)
        kmedoids_pd['predictions'] = np.concatenate((self.experiment.y[known_idx], y_pred), axis=0)

        known_train_pd = kmedoids_pd.copy()
        known_train_idx = np.concatenate((known_idx, train_idx), axis=0)
        known_train_pd['labels'] = self.experiment.y[known_train_idx]
        known_train_pd["idx"] = known_train_idx

        # Find the clusters and their centroids
        clusters, centroids, dist_matrix = run_kmedoids(kmedoids_pd,
                                                        n_clusters=self.no_clusters,
                                                        use_labels=self.use_labels,
                                                        use_weights=self.use_weights,
                                                        path=self.path,
                                                        plots_off=self.plots_off,
                                                        use_gower=use_gower)
        # Find the index of the query to be labeled
        wrong_points, query_idx = Annotator().select_from_worst_cluster(known_train_pd,
                                                                        clusters,
                                                                        train_idx,
                                                                        theta=theta,
                                                                        rng=self.experiment.rng,
                                                                        file=self.file,
                                                                        dist_matrix=dist_matrix)
        key = "xgl_" + str(theta)
        if not len(wrong_points) and key not in self.annotated_point.keys():
            self.annotated_point[key] = iteration

        # Plot the wrong points
        if len(wrong_points):
            run_kmedoids(kmedoids_pd,
                         n_clusters=self.no_clusters,
                         other_points=wrong_points,
                         use_labels=self.use_labels,
                         use_weights=self.use_weights,
                         path=self.path,
                         plots_off=self.plots_off,
                         use_gower=use_gower)

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
        theta = kwargs.get("param")

        X_train = get_from_indexes(self.experiment.X, train_idx)
        X_known = get_from_indexes(self.experiment.X, known_idx)
        X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)

        X_known_train = np.concatenate((X_known_norm, X_train_norm), axis=0)
        y_predicted = self.experiment.model.predict(X_known_train)

        column_names = self.experiment.feature_names
        if X_known_train.shape[1] != len(column_names):
            column_names = ['Col_' + str(i) for i in range(0, X_known_train.shape[1])]

        # Create a dataframe holding the known and train data with their predictions, labels and indexes in X
        X_known_train_pd = pd.DataFrame(data=X_known_train, columns=column_names)
        X_known_train_pd['is_train'] = np.concatenate([[False] * len(known_idx), [True] * len(train_idx)])
        X_known_train_pd['predictions'] = y_predicted
        X_known_train_pd['labels'] = np.concatenate([self.experiment.y[known_idx], self.experiment.y[train_idx]])
        X_known_train_pd["idx_in_X"] = np.concatenate([known_idx, train_idx])
        X_train_pd = X_known_train_pd[X_known_train_pd["is_train"]]

        X_known_train_features = X_known_train_pd.drop(['is_train', 'predictions', 'labels', 'idx_in_X'], axis=1)

        # Generate extra points
        if isinstance(self.experiment, Synthetic):
            extra_points_pd = self.generate_points(X_known_train_features)
            X_kte_pd = X_known_train_pd.append(extra_points_pd, sort=False)
            X_kte_features = X_kte_pd[column_names]
            kte_predictions = X_kte_pd['predictions']
        else:
            X_kte_features = X_known_train_features
            kte_predictions = X_known_train_pd['predictions']

        n_estimators = [5, 15, 30]
        num_features = len(self.experiment.feature_names)
        max_depth = num_features * 2 if num_features < 5 else num_features
        for n_estim in n_estimators:
            clf = SkopeRules(n_estimators=n_estim,
                             precision_min=0.4,
                             recall_min=0.01,
                             max_depth=max_depth,
                             max_features=None,
                             max_samples=1.0,
                             random_state=self.experiment.rng,
                             feature_names=column_names)

            clf.fit(X_kte_features, kte_predictions)
            if self.compare_performance(clf, 0.15, **kwargs):
                break

        sorted_rules = self.extract_rules(X_known_train_pd, clf)
        wrong_points_idx = []
        while len(wrong_points_idx) == 0:
            if len(sorted_rules) == 0:
                print("Selecting at random")
                return select_random(train_idx, self.experiment.rng)

            # Get the worst rule from the remaining rules
            worst_rule = self.select_rule(sorted_rules, theta)

            if hierarchy:
                # Find the points that satisfy the chosen rule
                points_known_train_pd = self.get_points_satisfying_rule(X_known_train_pd, worst_rule[0])
                kt_predictions = points_known_train_pd.predictions.to_numpy()
                if isinstance(self.experiment, Synthetic):
                    # Sample new points
                    extra_points_pd = self.generate_points_satisfying_rule(points_known_train_pd, worst_rule[0])
                    # Append them to points_known_train_pd and kt_predictions
                    points_known_train_pd = points_known_train_pd.append(extra_points_pd, sort=False)
                    kt_predictions = np.concatenate((kt_predictions, extra_points_pd.predictions.to_numpy()), axis=0)

                # Check if all predictions are the same
                if not all(element == kt_predictions[0] for element in kt_predictions):
                    # Get worst rule from the new points
                    self.file.write("Computing hierarchical rules...")
                    sorted_rules_hierarchy = self.extract_rules(points_known_train_pd, clf)
                    if len(sorted_rules_hierarchy) == 0:
                        print("Selecting at random in hierarchy")
                        return select_random(train_idx, self.experiment.rng)
                    worst_rule = self.select_rule(sorted_rules_hierarchy, theta)
                    X_train_pd = points_known_train_pd[points_known_train_pd["is_train"]]

            # Find the points that satisfy the chosen rule
            points_pd = self.get_points_satisfying_rule(X_train_pd, worst_rule[0])
            # From those, find the ones that are wrongly classified wrt the rules
            points_predictions = np.ones(len(points_pd)) * worst_rule[1]
            wrong_points_idx = points_pd[points_pd.labels != points_predictions]["idx_in_X"]

        return int(select_random(wrong_points_idx, self.experiment.rng))

    def xgl_rules_hierarchy(self, **kwargs):
        """
        Find index of the query to be labeled with the hierarchical XGL strategy using global surrogate model.

        :param kwargs: Keyword arguments

        :return: The index of the query (in X) to be labeled
        """
        return self.xgl_rules(**kwargs, hierarchy=True)

    ############## HELPER METHODS ##############
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

    def get_margins(self, data):
        if hasattr(self.experiment.model.sklearn_model, "decision_function"):
            margins = np.abs(self.experiment.model.decision_function(data))
        elif hasattr(self.experiment.model.sklearn_model, "predict_proba"):
            probs = self.experiment.model.predict_proba(data)
            margins = np.sum(probs * np.log(probs), axis=1).ravel()
        else:
            raise AttributeError("Model with either decision_function or predict_proba method")

        return margins

    def generate_points(self, points):
        # 1. Generate new points
        x_min, x_max = points.x.min(), points.x.max()
        y_min, y_max = points.y.min(), points.y.max()
        h_x = (x_max - x_min) / 100
        h_y = (y_max - y_min) / 100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))
        extra_points = np.c_[xx.ravel(), yy.ravel()]
        extra_points_pd = pd.DataFrame(data=extra_points, columns=self.experiment.feature_names)

        # 2. Use the blackbox predictor to predict their labels
        extra_points_pd['predictions'] = self.experiment.model.predict(extra_points)
        extra_points_pd['is_train'] = [False] * len(extra_points_pd)

        return extra_points_pd

    def generate_points_satisfying_rule(self, points, rule):
        extra_points_pd = self.generate_points(points)

        # to check if there are common points: pd.merge(points[['x', 'y']], extra_points_pd[['x', 'y']], how='inner')
        assert self.get_points_satisfying_rule(extra_points_pd, rule).equals(extra_points_pd)

        return extra_points_pd

    def get_data(self, **kwargs):
        train_idx = kwargs.get("train_idx")
        known_idx = kwargs.get("known_idx")
        test_idx = kwargs.get("test_idx")

        X_known, y_known = get_from_indexes(self.experiment.X, known_idx), self.experiment.y[known_idx]
        X_train, y_train = get_from_indexes(self.experiment.X, train_idx), self.experiment.y[train_idx]
        X_test, y_test = get_from_indexes(self.experiment.X, test_idx), self.experiment.y[test_idx]

        return X_known, y_known, X_train, y_train, X_test, y_test

    def compare_performance(self, clf, threshold, **kwargs):
        X_known, y_known, X_train, y_train, _, _ = self.get_data(**kwargs)
        # Normalize the data
        X_known, X_train = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)

        # Calculate accuracy wrt the extracted rules
        y_pred_train_rules = clf.predict(X_train)
        score_rules = self.calc_acc(y_train, y_pred_train_rules, "f1")

        # Get accuracy of the blackbox classifier
        score_blackbox = self.scores_f1[-1]

        return np.abs(score_blackbox - score_rules) < threshold

    def select_rule(self, rules, theta):
        logits = [1-x[2] for x in rules]
        exps = [np.exp(i * theta - max(logits)) for i in logits]
        softmax = [j / sum(exps) for j in exps]
        selected_rule_idx = self.experiment.rng.choice(len(rules), p=softmax)

        selected_rule = rules[selected_rule_idx]
        del rules[selected_rule_idx]

        return selected_rule

    def extract_rules(self, data, clf):
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
                plot_rules(clf, X_known_train_features.to_numpy(), predictions == idx, idx, self.path)

        self.file.write("Parameters for Skope Rules: {}".format(clf.get_params()))
        self.file.write("Number of extracted rules: {}". format(len(rules)))

        # Sort by the f1_score wrt the rules
        rules.sort(key=lambda x: x[2], reverse=True)

        return rules

    @staticmethod
    def get_points_satisfying_rule(X, rule):
        return X.query(rule)

    def run(self, method, known_idx, train_idx, test_idx):
        """
        Perform the learning loop: train a model, select a query and retrain the model until the budget is exhausted.

        :param method: The method to be used for query selection
        :param known_idx: The indexes of the known set
        :param train_idx: The indexes of the train set
        :param test_idx: The indexes of the test set

        :return: List of accuracy scores (F1 and AUC) on train and test set
        """
        # Empty the results arrays before every run
        self.query_array = []
        self.scores_f1 = []
        self.test_scores_f1 = []
        self.scores_auc = []
        self.test_scores_auc = []
        self.query_scores = []
        self.cos_distance_matrix = None

        # Get the theta value for XGL and rules or beta value for density based AL
        param = ""
        if "xgl" in method or "rules" in method or "density" in method:
            param = float(method.split("_")[-1])
            method = "_".join(method.split("_")[:-1])

        # 1. Train a model
        y_pred, y_pred_test = self.train_and_get_acc(known_idx, train_idx, test_idx)

        for iteration in tqdm(range(self.max_iter)):
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
                                             param=param,
                                             use_clustering=False)

            if query_idx is None:
                continue
            self.file.write("Selected point: {}\n".format(get_from_indexes(self.experiment.X, query_idx)))

            if not self.plots_off:
                plot_decision_surface(self.experiment,
                                      known_idx,
                                      train_idx,
                                      query_idx=query_idx,
                                      title=str(iteration) + " " + self.experiment.model.name + " " + method,
                                      path=self.path)

            # 3. Query an "oracle" and add the labeled example to the training set
            known_idx, train_idx = self.move(known_idx, train_idx, query_idx)

            # 4. Retrain the model with the new training set
            y_pred, y_pred_test = self.train_and_get_acc(known_idx, train_idx, test_idx, query_idx)

        # Plot the decision surface
        if not self.plots_off:
            plot_decision_surface(self.experiment,
                                  known_idx,
                                  train_idx,
                                  title="Last iteration " + self.experiment.model.name + " " + method,
                                  path=self.path)
            # Plot the predictions
            plot_decision_surface(self.experiment,
                                  known_idx,
                                  test_idx,
                                  y_pred=y_pred_test,
                                  title="Predictions " + self.experiment.model.name + " " + method,
                                  path=self.path)

