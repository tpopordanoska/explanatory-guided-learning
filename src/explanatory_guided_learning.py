import numpy as np
from scipy.spatial import distance

from src import select_random


class Annotator:
    """
    Class containing methods for the explanatory_guided query selection strategy.
    """
    def select_from_worst_cluster(self, pd_points, clusters, train_idx, theta, rng, file=None):
        """
        Select index of the point to be labeled from the cluster containing largest number of wrong points.

        :param pd_points: Pandas DataFrame containing the features, the predictions and the true labels of the points
        :param clusters: The clusters (indexes of the points) found with kmedoids
        :param train_idx: The indexes of the train dataset (unlabeled points)
        :param theta: Parameter for the softmax function
        :param rng: RandomState object
        :param file: The output file

        :return: The wrongly classified points, the index of the closest wrong instance (in X)
        """
        # Find all the wrongly classified examples (model predictions vs true labels)
        wrong_points, clusters_lookup, lookup = find_wrong_points(pd_points, clusters)
        file.write("Number of wrong points: {}\n".format(len(wrong_points)))
        if not len(wrong_points):
            # If there are no more wrongly classified points, proceed with random sampling from train
            query_idx = select_random(train_idx, rng)
            return [], query_idx
        # Find the cluster with the most wrongly classified examples
        # max_key, max_value = max(lookup.items(), key=lambda x: len(x[1]))

        # Softmax
        # Using the number of wrongly classified points
        logits = [len(x) for x in lookup.values()]
        file.write("Logits using plain numbers")
        # Using a ratio of wrong points to the total number of points in known+train
        # logits = [len(x)/(pd_points.shape[0]) for x in lookup.values()]
        # file.write("Logits using ratio")

        exps = [np.exp(i * theta - max(logits)) for i in logits]
        softmax = [j / sum(exps) for j in exps]
        selected_cluster_key = rng.choice(list(lookup.keys()), p=softmax)
        selected_cluster_value = lookup.get(selected_cluster_key)
        if file is not None:
            file.write("Logits: {}, exps: {}, softmax: {}\n".format(logits, exps, softmax))
            file.write("Cluster lookup keys: {}\n".format(lookup.keys()))
            file.write("Selected_cluster_key: {}, value: {}\n".format(selected_cluster_key, selected_cluster_value))

        # Find the centroid of that custer, it's the first element in the list
        max_centroid = clusters_lookup[selected_cluster_key][0]
        # Drop the labels and predictions columns to measure distance
        pd_points_features = pd_points.copy()
        pd_points_features = pd_points_features.drop(columns=["labels", "predictions", "idx"])
        # Find the element with min proximity to the centroid for each wrongly classified point in that cluster
        closest_wrong_idx = min(selected_cluster_value, key=lambda x: distance.euclidean(
            pd_points_features.iloc[x], pd_points_features.iloc[max_centroid]))

        query_idx = int(pd_points.iloc[closest_wrong_idx].idx)
        return wrong_points, query_idx

    def select_closest(self, pd_points, clusters):
        """
        Select index of the point which lies closest to a centroid.

        :param pd_points: Pandas DataFrame containing the features, the predictions and the true labels of the points
        :param clusters:
        :return:
        """
        wrong_points, clusters_lookup, lookup = find_wrong_points(pd_points, clusters)
        if len(wrong_points) == 0:
            return [], None
        # Find the proximity to the closest centroid found with k-medoids for each wrongly classified point
        distances = dist_to_centroid(lookup, clusters_lookup, pd_points)

        closest_wrong_idx = min(distances, key=distances.get)
        # Sort them and return the wrongly classified example closest to a centorid
        query_idx = int(pd_points.iloc[closest_wrong_idx].idx)
        return wrong_points, query_idx


def find_wrong_points(pd_points, clusters):
    """
    Find wrongly classified points in the given data.

    :param pd_points: The data
    :param clusters: The clusters

    :return: The wrongly classified points, lookup for the clusters, and lookup for the wrong points
    """
    # Find the points where predictions are wrong
    wrong_points = pd_points[pd_points.labels != pd_points.predictions]
    clusters_lookup = create_clusters_lookup(clusters)
    lookup = create_lookup(wrong_points.index, clusters_lookup)

    return wrong_points, clusters_lookup, lookup


def create_clusters_lookup(clusters):
    """
    Create lookup for the clusters: the value for each key is an array of indexes of the points in the cluster.

    :param clusters: Clusters represented as arrays of indexes of points

    :return: The lookup
    """
    return {i: cluster for i, cluster in enumerate(clusters)}


def create_lookup(points, clusters_lookup):
    """
    Returns dictionary which has as keys the keys from the cluster_lookup, and the values are the wrong points in the
    corresponding cluster

    :param points: The data
    :param clusters_lookup: The lookup for the clusters

    :return: The lookup for the wrong points
    """
    lookup = {}
    for point in points:
        for key, value in clusters_lookup.items():
            if point in value:
                if key not in lookup:
                    lookup[key] = [point]
                    break
                else:
                    lookup[key].append(point)
                    break
    return lookup


def dist_to_centroid(lookup, clusters_lookup, pd_points):
    """
    Calculate distance from each wrongly classified point to the centroid of the cluster that it belongs to.

    :param lookup: Lookup for the wrong points
    :param clusters_lookup: Lookup for the clusters
    :param pd_points: The data

    :return: A dictionary containing the calculated distances for each wrongly classified point to the centroid
    """
    distances = {}
    # Remove the unnecessary columns (labels and predictions)
    pd_points_features = pd_points.drop(columns=["labels", "predictions", "idx"])
    for (key, value) in lookup.items():
        for element in value:
            centroid = clusters_lookup[key][0]
            # Calculate euclidean distance on the features
            distances[element] = distance.euclidean(pd_points_features.iloc[element], pd_points_features.iloc[centroid])
    return distances


