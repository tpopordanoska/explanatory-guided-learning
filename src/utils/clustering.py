import gower
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids

from src.utils.plotting import plot_kmedoids


def introduce_uu(experiment, weight):
    """
    Introduce unknown unknowns by flipping the class of random sub-groups from the training data

    :param experiment: The experiment being run
    :param weight: The weight of the unknown unknowns
    """
    X_y = np.concatenate((experiment.X, experiment.y[:, None]), axis=1)
    kmedoids_instance, _ = get_kmedoids_instance(X_y, 100, experiment.use_gower)
    clusters = kmedoids_instance.get_clusters()

    clusters_to_flip = experiment.rng.random_integers(len(clusters), size=10)
    idx_to_flip = [item for cluster in clusters_to_flip for item in clusters[cluster]]
    for idx in idx_to_flip:
        experiment.y[idx] = toggle(experiment.y[idx])
        experiment.sample_weights[idx] *= weight
    print(f"UUs weight: {weight}")
    return


def toggle(value):
    return 1 - value


def run_kmedoids(points_pd, running_instance, other_points=None):
    """
    Run kmedoids algorithm on the given points with the given number of clusters and plot the centroids.

    :param points_pd: Pandas DataFrame containing the features and the predicted labels on the train set
    :param other_points: Other points (ex. wrongly classified) to plot instead of the points used to find clusters
    :param running_instance: An object holding information about the current experiment, args and strategy

    :return: Indexes (in points) of the found clusters and the corresponding centroids
    """
    n_clusters = running_instance.args.n_clusters
    plots_on = running_instance.args.plots_on
    use_gower = running_instance.experiment.use_gower

    points = preprocess(points_pd)

    kmedoids_instance, matrix = get_kmedoids_instance(points, n_clusters, use_gower)
    centroids_idx = kmedoids_instance.get_medoids()

    if plots_on:
        plot_kmedoids(points, kmedoids_instance, other_points, running_instance.experiment.path)

    return kmedoids_instance.get_clusters(), centroids_idx, matrix


def preprocess(points_pd):
    weighted_points_pd = points_pd.copy()
    weighted_points_pd["predictions"] = points_pd["predictions"] * (len(points_pd.columns) - 1)

    return weighted_points_pd.to_numpy()


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
