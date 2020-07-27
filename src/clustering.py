import gower
from pyclustering.cluster.kmedoids import kmedoids

from .plotting import *


def introduce_uu(experiment):
    X_y = np.concatenate((experiment.X, experiment.y[:, None]), axis=1)
    kmedoids_instance, _ = get_kmedoids_instance(X_y, 100, experiment.use_gower)
    clusters = kmedoids_instance.get_clusters()

    clusters_to_flip = experiment.rng.random_integers(len(clusters), size=10)
    idx_to_flip = [item for cluster in clusters_to_flip for item in clusters[cluster]]
    for idx in idx_to_flip:
        experiment.y[idx] = toggle(experiment.y[idx])
    return


def toggle(value):
    return 1-value


def run_kmedoids(points_pd, n_clusters, other_points=None, use_labels=False,
                 use_weights=False, path=None, plots_on=False, use_gower=False):
    """
    Run kmedoids algorithm on the given points with the given number of clusters and plot the centroids.

    :param points_pd: Pandas DataFrame containing the features and the predicted labels on the train set
    :param n_clusters: The number of clusters to be found in the data
    :param other_points: Other points (ex. wrongly classified) to plot instead of the points used to find clusters
    :param use_labels: Whether to use the labels of the points as an attribute
    :param use_weights: Whether to weigh the labels by the number of other attributes
    :param path: The path to the folder where the graphs will be saved
    :param plots_on: Whether to plot the graphs
    :param use_gower: Whether to calculate distance using gower metric

    :return: Indexes (in points) of the found clusters and the corresponding centroids
    """
    points = preprocess(points_pd, use_labels, use_weights)

    kmedoids_instance, matrix = get_kmedoids_instance(points, n_clusters, use_gower)
    centroids_idx = kmedoids_instance.get_medoids()

    if plots_on:
        plot_kmedoids(points, kmedoids_instance, other_points, path)

    return kmedoids_instance.get_clusters(), centroids_idx, matrix


def preprocess(points_pd, use_labels, use_weights):
    # If we don't want to use predictions
    if not use_labels:
        points = points_pd.drop(columns=["predictions"]).to_numpy()
    # If we want to use predictions and weight them by the number of attributes
    elif use_weights:
        weighted_points_pd = points_pd.copy()
        weighted_points_pd["predictions"] = points_pd["predictions"] * (len(points_pd.columns) - 1)
        points = weighted_points_pd.to_numpy()
    # If we want to use the labels, but not weigh them
    else:
        points = points_pd.to_numpy()

    return points


def get_kmedoids_instance(points, n_clusters, use_gower=False):
    np.random.seed(0)
    initial_medoids = np.random.randint(len(points), size=n_clusters)

    if use_gower:
        matrix = gower.gower_matrix(points)
        kmedoids_instance = kmedoids(matrix, initial_medoids, data_type='distance_matrix')
    else:
        matrix = None
        kmedoids_instance = kmedoids(points, initial_medoids)

    return kmedoids_instance.process(), matrix

