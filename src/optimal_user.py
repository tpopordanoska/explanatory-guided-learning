import numpy as np
from scipy.spatial import distance


class Annotator:
    # The Annotator should know the unlabeled data (points with x,y coordinates) and their labels
    def __init__(self, points):
        self.points = points

    def select_from_worst_cluster(self, all_points, clusters):
        # Find all the wrongly classified examples (model predictions vs true labels)
        wrong_points, clusters_lookup, lookup = find_wrong_points(self.points, all_points, clusters)

        # Find the cluster with the most wrongly classified examples
        max_key, max_value = max(lookup.items(), key = lambda x: len(x[1]))

        # Find the centroid of that custer, it's the first element in the list
        max_centroid = clusters_lookup[max_key][0]
        # Find the element with min proximity to the centroid found with k-medoids for each wrongly classified
        # point in that cluster
        closest_wrong_idx = min(max_value, key=lambda x: distance.euclidean(all_points[x][0:2],
                                                                            all_points[max_centroid][0:2]))

        return wrong_points, closest_wrong_idx, all_points[closest_wrong_idx]

    def select_closest(self, all_points, clusters):
        wrong_points, clusters_lookup, lookup = find_wrong_points(self.points, all_points, clusters)
        if len(wrong_points) == 0:
            return [], None, []
        # Find the proximity to the closest centroid found with k-medoids for each wrongly classified point
        distances = dist_to_centroid(lookup, clusters_lookup, all_points)

        closest_wrong_idx = min(distances, key=distances.get)
        # Sort them and return the wrongly classified example closest to a centorid
        return wrong_points, closest_wrong_idx, all_points[closest_wrong_idx]
        # wrong_idx = np.where(points == wrong_points)


def create_clusters_lookup(clusters):
    return {i: cluster for i, cluster in enumerate(clusters)}


def create_lookup(points, clusters_lookup):
    lookup = {}
    for point in points:
        for key, value in clusters_lookup.items():
            if point[0][0] in value:
                if key not in lookup:
                    lookup[key] = [point[0][0]]
                else:
                    lookup[key].append(point[0][0])
    return lookup


def dist_to_centroid(lookup, clusters_lookup, all_points):
    distances = {}
    for (key, value) in lookup.items():
        for element in value:
            centroid = clusters_lookup[key][0]
            distances[element] = distance.euclidean(all_points[element][0:2], all_points[centroid][0:2])
    return distances


def find_wrong_points(points, all_points, clusters):
    wrong_points = points[points[:, 2] != points[:, 3]]
    indexes = [np.where((all_points[:, 0] == wrong_point[0]) & (all_points[:, 1] == wrong_point[1]))
               for wrong_point in wrong_points]
    clusters_lookup = create_clusters_lookup(clusters)
    lookup = create_lookup(indexes, clusters_lookup)

    return wrong_points, clusters_lookup, lookup