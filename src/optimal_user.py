import numpy as np
from scipy.spatial import distance


class Annotator:
    # The Annotator should know the training data data (points with x,y coordinates) and their labels
    def __init__(self, points):
        self.points = points

    def select_from_worst_cluster(self, pd_points, clusters, theta=1, rng=0):
        """

        :param pd_points: Matrix containing the features, the predictions and the true labels
        :param clusters: The clusters found with kmedoids
        :return: The wrongly classified points, the index of the closest wrong instance in all_points and the closest wrong instance itself

        """
        # Find all the wrongly classified examples (model predictions vs true labels)
        wrong_points, clusters_lookup, lookup = self.find_wrong_points(pd_points, clusters)
        if len(wrong_points) == 0:
            return [], None, []
        # Find the cluster with the most wrongly classified examples
        # max_key, max_value = max(lookup.items(), key=lambda x: len(x[1]))

        # Softmax
        logits = [len(x) for x in lookup.values()]
        exps = [np.exp(i * theta) for i in logits]
        softmax = [j / sum(exps) for j in exps]
        selected_cluster_key = rng.choice(list(lookup.keys()), p=softmax)
        selected_cluster_value = lookup.get(selected_cluster_key)

        # Find the centroid of that custer, it's the first element in the list
        max_centroid = clusters_lookup[selected_cluster_key][0]
        # Drop the labels and predictions columns to measure distance
        pd_points_features = pd_points.drop(columns=["labels", "predictions"])
        # Find the element with min proximity to the centroid found with k-medoids for each wrongly classified
        # point in that cluster
        closest_wrong_idx = min(selected_cluster_value, key=lambda x: distance.euclidean(pd_points_features.iloc[x],
                                                                            pd_points_features.iloc[max_centroid]))

        return wrong_points, closest_wrong_idx, pd_points.iloc[closest_wrong_idx]

    def select_closest(self, pd_points, clusters):
        wrong_points, clusters_lookup, lookup = self.find_wrong_points(pd_points, clusters)
        if len(wrong_points) == 0:
            return [], None, []
        # Find the proximity to the closest centroid found with k-medoids for each wrongly classified point
        distances = dist_to_centroid(lookup, clusters_lookup, pd_points)

        closest_wrong_idx = min(distances, key=distances.get)
        # Sort them and return the wrongly classified example closest to a centorid
        return wrong_points, closest_wrong_idx, pd_points.iloc[closest_wrong_idx]
        # wrong_idx = np.where(points == wrong_points)

    def find_wrong_points(self, pd_all_points, clusters):
        # Find the points where predictions are wrong
        wrong_points = pd_all_points[pd_all_points.labels != pd_all_points.predictions]
        clusters_lookup = create_clusters_lookup(clusters)
        lookup = create_lookup(wrong_points.index, clusters_lookup)

        return wrong_points, clusters_lookup, lookup


def create_clusters_lookup(clusters):
    return {i: cluster for i, cluster in enumerate(clusters)}


def create_lookup(points, clusters_lookup):
    """
    Returns dictionary which has as keys the keys from the cluster_lookup, and the values are the wrong points in the
    corresponding cluster

    :param points:
    :param clusters_lookup:
    :return:
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
    distances = {}
    # Remove the last two columns (lables and predictions)
    pd_points_features = pd_points.drop(columns=["labels", "predictions"])
    for (key, value) in lookup.items():
        for element in value:
            centroid = clusters_lookup[key][0]
            # Calculate euclidean distance on the features
            distances[element] = distance.euclidean(pd_points_features.iloc[element], pd_points_features.iloc[centroid])
    return distances


