import numpy as np
import pandas as pd
from skrules import SkopeRules
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

from src.experiments import Synthetic
from src.running_instance import RunningInstance
from src.utils.normalizer import Normalizer
from src.utils.plotting import create_meshgrid, plot_rules


class XglRules(RunningInstance):

    def __init__(self, hierarchy=False, simple_tree=False, **kwargs):
        self.hierarchy = hierarchy
        self.simple_tree = simple_tree
        super().__init__(**kwargs)

    def query(self):
        X_known_train = self.get_normalized_known_train_features()
        column_names = self.get_column_names(X_known_train)

        X_known_train_pd = self.create_dataframe(X_known_train, column_names)
        X_train_pd = X_known_train_pd[X_known_train_pd["is_train"]]

        X_kte_features, kte_predictions = self.prepare_data(X_known_train_pd)
        if isinstance(self.experiment, Synthetic):
            X_kte_features, kte_predictions = self.prepare_and_generate_extra_data(X_known_train_pd, column_names)

        if self.simple_tree:
            clf = DecisionTreeClassifier()
            clf.fit(self.get_feature_columns(X_known_train_pd), X_known_train_pd['predictions'])
            self.save_rules_acc(X_known_train, clf)
            sorted_rules = self.extract_rules_from_simple_tree(X_known_train_pd, clf, column_names)
        else:
            clf = self.find_clf_params(X_kte_features, kte_predictions, column_names)
            self.save_rules_acc(X_known_train, clf)
            sorted_rules = self.extract_rules(X_known_train_pd, clf)

        wrong_points_idx = []
        while len(wrong_points_idx) == 0:
            if len(sorted_rules) == 0:
                self.experiment.file.write("Selecting at random")
                return self.select_random(self.train_idx, self.experiment.rng)

            # Get the worst rule from the remaining rules
            worst_rule = self.select_and_remove_rule(sorted_rules, self.param)

            if self.hierarchy:
                points_kt_pd, kt_predictions = self.get_data_satisfying_rules(X_known_train_pd, worst_rule)
                if isinstance(self.experiment, Synthetic):
                    points_kt_pd, kt_predictions = self.sample_extra_points_for_synthetic(points_kt_pd, kt_predictions, worst_rule)
                # Check if all predictions are the same
                if not all(element == kt_predictions[0] for element in kt_predictions):
                    sorted_rules_hierarchy = self.compute_hierarchical_rules(points_kt_pd, clf)
                    if len(sorted_rules_hierarchy) == 0:
                        self.experiment.file.write("Selecting at random in hierarchy \n")
                        return self.select_random(self.train_idx, self.experiment.rng)
                    worst_rule = self.select_and_remove_rule(sorted_rules_hierarchy, self.param)
                    X_train_pd = points_kt_pd[points_kt_pd["is_train"]]

            points_pd = self.get_points_satisfying_rule(X_train_pd, worst_rule[0])
            wrong_points_idx = self.get_mistake_satisfying_rule(points_pd, worst_rule)

        query_idx = int(self.select_random(wrong_points_idx, self.experiment.rng))
        self.count_false_mistakes(points_pd, query_idx)

        return query_idx

    def get_normalized_known_train_features(self):
        X_known, X_train = self.get_known_train_features()
        X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)

        return np.concatenate((X_known_norm, X_train_norm), axis=0)

    def get_column_names(self, X_known_train):
        column_names = self.experiment.feature_names
        if X_known_train.shape[1] != len(column_names):
            column_names = ['Col_' + str(i) for i in range(0, X_known_train.shape[1])]

        return column_names

    def create_dataframe(self, X_known_train, column_names):
        y_predicted = self.predict(X_known_train)
        # Create a dataframe holding the known and train data with their predictions, labels and indexes in X
        X_known_train_pd = pd.DataFrame(data=X_known_train, columns=column_names)
        X_known_train_pd['is_train'] = np.concatenate([[False] * len(self.known_idx), [True] * len(self.train_idx)])
        X_known_train_pd['predictions'] = y_predicted
        X_known_train_pd['labels'] = np.concatenate([self.experiment.y[self.known_idx], self.experiment.y[self.train_idx]])
        X_known_train_pd["idx_in_X"] = np.concatenate([self.known_idx, self.train_idx])

        return X_known_train_pd

    def prepare_data(self, X_known_train_pd):
        X_known_train_features_pd = self.get_feature_columns(X_known_train_pd)
        X_kte_features = X_known_train_features_pd
        kte_predictions = X_known_train_pd['predictions']

        return X_kte_features, kte_predictions

    def prepare_and_generate_extra_data(self, X_known_train_pd, column_names):
        X_known_train_features_pd = self.get_feature_columns(X_known_train_pd)
        extra_points_pd = self.generate_points(X_known_train_features_pd)
        X_kte_pd = X_known_train_pd.append(extra_points_pd, sort=False)
        X_kte_features = X_kte_pd[column_names]
        kte_predictions = X_kte_pd['predictions']

        return X_kte_features, kte_predictions

    def save_rules_acc(self, points, clf):
        """
        Save the faithfulness (f1 of rules w.r.t. blackbox) of the rules for the Synthetic experiment.

        :param points: The set of known and train points
        :param clf: SkopeRules classifier
        """
        if isinstance(self.experiment, Synthetic):
            points = self.sample_extra_between(points)
        pred_rules = clf.predict(points)
        pred_blackbox = self.predict(points)
        score_rules_wrt_bb = self.get_f1_score(pred_blackbox, pred_rules)
        self.results.rules_wrt_blackbox_f1.append(score_rules_wrt_bb)

    def find_clf_params(self, X_kte_features, kte_predictions, column_names):
        clf = SkopeRules()
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
            if self.compare_performance(clf, 0.15):
                break

        return clf

    def extract_rules(self, data, clf):
        X_known_train_features = self.get_feature_columns(data)
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
            if self.args.plots_on:
                plot_rules(clf, X_known_train_features.to_numpy(), predictions == idx,
                           idx, self.experiment.path, self.results.rules_wrt_blackbox_f1[-1])

        self.experiment.file.write("Parameters for Skope Rules: {} \n".format(clf.get_params()))
        self.experiment.file.write("Number of extracted rules: {} \n". format(len(rules)))

        # Sort by the f1_score wrt the rules
        rules.sort(key=lambda x: x[2], reverse=True)

        return rules

    def extract_rules_from_simple_tree(self, data, tree, feature_names):
        X_known_train_features = self.get_feature_columns(data)
        predictions = data['predictions']
        X_train_pd = data[data["is_train"]]

        final_rules = []
        tree.fit(X_known_train_features, predictions)
        rules_with_labels_from_tree = self.tree_to_rules(tree, feature_names)
        rules_tuples = [(r, l, self.eval_rule_perf(r, X_known_train_features, predictions)) for r, l in
                        set(rules_with_labels_from_tree)]
        for rule in rules_tuples:
            points_pd = self.get_points_satisfying_rule(X_train_pd, rule[0])
            # if there are no points in X_train that satisfy the rule, skip it
            if len(points_pd) > 0:
                points_predictions = np.ones(len(points_pd)) * rule[1]
                score = self.get_f1_score(points_pd.labels, points_predictions)
                score_blackbox = self.get_f1_score(points_pd.labels, points_pd.predictions)
                # Each element in rules is a tuple (rule, class, f1_score wrt rules, f1_score wrt classifier)
                final_rules.append((rule[0], rule[1], score, score_blackbox))

        if self.args.plots_on:
            plot_rules(tree, X_known_train_features.to_numpy(), predictions, "",
                       self.experiment.path, self.results.rules_wrt_blackbox_f1[-1])

        # Sort by the f1_score wrt the rules
        final_rules.sort(key=lambda x: x[2], reverse=True)

        return final_rules

    def select_and_remove_rule(self, rules, theta):
        logits = [1-x[2] for x in rules]
        exps = [np.exp(i * theta - max(logits)) for i in logits]
        softmax = [j / sum(exps) for j in exps]
        selected_rule_idx = self.experiment.rng.choice(len(rules), p=softmax)

        selected_rule = rules[selected_rule_idx]
        del rules[selected_rule_idx]

        return selected_rule

    def get_data_satisfying_rules(self, X_known_train_pd, worst_rule):
        points_known_train_pd = self.get_points_satisfying_rule(X_known_train_pd, worst_rule[0])
        kt_predictions = points_known_train_pd.predictions.to_numpy()

        return points_known_train_pd, kt_predictions

    def sample_extra_points_for_synthetic(self, points_known_train_pd, kt_predictions, worst_rule):
        extra_points_pd = self.generate_points_satisfying_rule(points_known_train_pd, worst_rule[0])
        # Append them to points_known_train_pd and kt_predictions
        points_known_train_pd = points_known_train_pd.append(extra_points_pd, sort=False)
        kt_predictions = np.concatenate((kt_predictions, extra_points_pd.predictions.to_numpy()), axis=0)

        return points_known_train_pd, kt_predictions

    def compute_hierarchical_rules(self, points_kt_pd, clf):
        self.experiment.file.write("Computing hierarchical rules... \n")
        return self.extract_rules(points_kt_pd, clf)

    @staticmethod
    def get_mistake_satisfying_rule(points_pd, worst_rule):
        # Find the points that are wrongly classified wrt the rules
        points_predictions = np.ones(len(points_pd)) * worst_rule[1]
        wrong_points_idx = points_pd[points_pd.labels != points_predictions]["idx_in_X"]

        return wrong_points_idx

    @staticmethod
    def get_feature_columns(X):
        return X.drop(['is_train', 'predictions', 'labels', 'idx_in_X'], axis=1)

    def count_false_mistakes(self, points_pd, query_idx):
        # Count false mistakes = wrongly classified point wrt rules but blackbox prediction is correct
        selected_point = points_pd[points_pd["idx_in_X"] == query_idx]
        is_false_mistake = int((selected_point.labels == selected_point.predictions).bool())
        self.results.false_mistakes_count.append(is_false_mistake + self.results.false_mistakes_count[-1])

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
        extra_points_pd['predictions'] = self.predict(extra_points)
        extra_points_pd['is_train'] = [False] * len(extra_points_pd)

        return extra_points_pd

    def generate_points_satisfying_rule(self, points, rule):
        extra_points_pd = self.generate_points(points)

        # to check if there are common points: pd.merge(points[['x', 'y']], extra_points_pd[['x', 'y']], how='inner')
        assert self.get_points_satisfying_rule(extra_points_pd, rule).equals(extra_points_pd)

        return extra_points_pd

    def compare_performance(self, clf, threshold):
        X_known, y_known, X_train, y_train, _, _ = self.get_all_data()
        # Normalize the data
        X_known, X_train = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)

        # Calculate accuracy wrt the extracted rules
        y_pred_train_rules = clf.predict(X_train)
        score_rules = self.calc_acc(y_train, y_pred_train_rules, "f1")

        # Get accuracy of the blackbox classifier
        score_blackbox = self.results.scores_f1[-1]

        return np.abs(score_blackbox - score_rules) < threshold

    @staticmethod
    def get_points_satisfying_rule(X, rule):
        return X.query(rule)

    @staticmethod
    def sample_extra_between(points):
        xx, yy = create_meshgrid(points, 0.01)
        return np.c_[xx.ravel(), yy.ravel()]

    @staticmethod
    # from https://github.com/scikit-learn-contrib/skope-rules/tree/master/skrules
    def eval_rule_perf(rule, X, y):
        detected_index = list(X.query(rule).index)
        if len(detected_index) <= 1:
            return (0, 0)
        y_detected = y[detected_index]
        true_pos = y_detected[y_detected > 0].sum()
        if true_pos == 0:
            return (0, 0)
        pos = y[y > 0].sum()
        return y_detected.mean(), float(true_pos) / pos

    @staticmethod
    # from https://github.com/scikit-learn-contrib/skope-rules/tree/master/skrules
    def tree_to_rules(tree, feature_names):
        """
        Return a list of rules from a tree
        """
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature]
        rules = []

        def recurse(node, base_name):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                symbol = '<='
                symbol2 = '>'
                threshold = tree_.threshold[node]
                text = base_name + ["{} {} {}".format(name, symbol, threshold)]
                recurse(tree_.children_left[node], text)

                text = base_name + ["{} {} {}".format(name, symbol2,
                                                      threshold)]
                recurse(tree_.children_right[node], text)
            else:
                rule = str.join(' and ', base_name)
                rule = (rule if rule != '' else ' == '.join([feature_names[0]] * 2))
                label = tree.classes_[np.argmax(tree_.value[node])] * 1
                rules.append((rule, label))

        recurse(0, [])

        return rules if len(rules) > 0 else 'True'
