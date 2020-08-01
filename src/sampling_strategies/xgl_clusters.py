from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import distance

from src.running_instance import RunningInstance
from src.utils.clustering import run_kmedoids
from src.utils.normalizer import Normalizer


class XglClusters(RunningInstance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self):
        X_known, X_train = self.get_known_train_features()
        X_known_train, y_pred = self.get_transformed_data(X_known, X_train)

        kmedoids_pd, known_train_pd = self.create_dataframes(X_known_train, y_pred)
        clusters, centroids, dist_matrix = run_kmedoids(kmedoids_pd, self)
        wrong_points, query_idx = self.select_from_worst_cluster(known_train_pd, clusters, dist_matrix)

        self.mark_iteration_if_no_mistakes(wrong_points)

        # Plot the wrong points
        if len(wrong_points):
            run_kmedoids(kmedoids_pd, self, other_points=wrong_points)

        return query_idx

    def get_transformed_data(self, X_known, X_train):
        if self.experiment.use_gower:
            # No need to normalize the data, as it is done in the gower_matrix() method
            X_known_train = np.concatenate((X_known, X_train), axis=0)
            y_pred = self.normalize_and_predict(X_train)
        else:
            X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)
            X_known_train = np.concatenate((X_known_norm, X_train_norm), axis=0)
            y_pred = self.predict(X_train_norm)

        return X_known_train, y_pred

    def create_dataframes(self, X_known_train, y_pred):
        kmedoids_pd = pd.DataFrame(data=X_known_train)
        kmedoids_pd['predictions'] = np.concatenate((self.experiment.y[self.known_idx], y_pred), axis=0)

        known_train_pd = kmedoids_pd.copy()
        known_train_idx = np.concatenate((self.known_idx, self.train_idx), axis=0)
        known_train_pd['labels'] = self.experiment.y[known_train_idx]
        known_train_pd["idx"] = known_train_idx

        return kmedoids_pd, known_train_pd

    def mark_iteration_if_no_mistakes(self, wrong_points):
        key = "xgl_" + str(self.param)
        # Mark the first time no mistakes are found
        if not len(wrong_points) and key not in self.annotated_point.keys():
            self.annotated_point[key] = self.iteration

    def select_from_worst_cluster(self, pd_points, clusters, dist_matrix=None):
        """
        Find index of the point to be labeled from the cluster containing largest number of wrong points.

        :param pd_points: Pandas DataFrame containing the features, the predictions and the true labels of the points
        :param clusters: The clusters (indexes of the points) found with kmedoids
        :param dist_matrix: Matrix of distances calculated with cosine similarity

        :return: The wrongly classified points, the index of the closest wrong instance (in X)
        """
        wrong_points, clusters_lookup, wrong_points_per_cluster = self.find_wrong_points(pd_points, clusters)
        if not len(wrong_points):
            # If there are no more wrongly classified points, proceed with random sampling from train
            query_idx = self.select_random(self.train_idx, self.experiment.rng)
            return [], query_idx

        softmax = self.calculate_softmax(wrong_points_per_cluster)
        selected_cluster_key, selected_cluster = self.select_cluster(wrong_points_per_cluster, softmax)

        # Find the centroid of that custer, it's the first element in the list
        max_centroid = clusters_lookup[selected_cluster_key][0]
        closest_wrong_idx = self.find_closest_to_centroid(max_centroid, dist_matrix, pd_points, selected_cluster)

        query_idx = int(pd_points.iloc[closest_wrong_idx].idx)
        return wrong_points, query_idx

    def find_wrong_points(self, pd_points, clusters):
        """
        Find wrongly classified examples (model predictions vs true labels)

        :param pd_points: The data
        :param clusters: The clusters

        :return: The wrongly classified points, lookup for the clusters, and lookup for the wrong points
        """
        # Find the points where predictions are wrong
        wrong_points = pd_points[pd_points.labels != pd_points.predictions]
        clusters_lookup = {i: cluster for i, cluster in enumerate(clusters)}
        lookup = self.create_lookup(wrong_points.index, clusters_lookup)
        self.experiment.file.write("Number of wrong points: {}\n".format(len(wrong_points)))

        return wrong_points, clusters_lookup, lookup

    def calculate_softmax(self, lookup):
        # Using the number of wrongly classified points
        logits = [len(x) for x in lookup.values()]
        self.experiment.file.write("Logits using plain numbers")
        # Using a ratio of wrong points to the total number of points in known+train
        # logits = [len(x)/(pd_points.shape[0]) for x in lookup.values()]
        # file.write("Logits using ratio")

        exps = [np.exp(i * self.param - max(logits)) for i in logits]
        softmax = [j / sum(exps) for j in exps]
        self.experiment.file.write("Logits: {}, exps: {}, softmax: {}\n".format(logits, exps, softmax))

        return softmax

    def select_cluster(self, lookup, softmax):
        selected_cluster_key = self.experiment.rng.choice(list(lookup.keys()), p=softmax)
        selected_cluster_value = lookup.get(selected_cluster_key)

        return selected_cluster_key, selected_cluster_value

    @staticmethod
    def find_closest_to_centroid(max_centroid, dist_matrix, pd_points, selected_cluster):
        # Drop the labels and predictions columns to measure distance
        pd_points_features = pd_points.copy()
        pd_points_features = pd_points_features.drop(columns=["labels", "predictions", "idx"])
        # Find the element with min proximity to the centroid for each wrongly classified point in that cluster
        if dist_matrix is not None:
            closest_wrong_idx = min(selected_cluster, key=lambda x: dist_matrix[max_centroid, x])
        else:
            closest_wrong_idx = min(selected_cluster, key=lambda x: distance.euclidean(
                pd_points_features.iloc[x], pd_points_features.iloc[max_centroid]))

        return closest_wrong_idx

    @staticmethod
    def create_lookup(points, clusters_lookup):
        """
        Returns dictionary which has as keys the keys from the cluster_lookup, and the values are the wrong points in the
        corresponding cluster

        :param points: The data
        :param clusters_lookup: The lookup for the clusters

        :return: The lookup for the wrong points
        """
        lookup = defaultdict(list)
        for point in points:
            for key, value in clusters_lookup.items():
                if point in value:
                    lookup[key].append(point)

        return lookup
