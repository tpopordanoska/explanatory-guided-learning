import matplotlib.pyplot as plt
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state


def run_kmeans(points, n_clusters=2, use_labels="True"):
    """
    Runs k-means algorithm on the given points with the given number of clusters and plot the centroids.

    :param points: The dataset.
    :param n_clusters: The number of clusters to be found in the data.
    :param use_labels: Whether to use the labels of the points as an attribute.

    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    if use_labels:
        kmeans.fit_predict(points)
    else:
        kmeans.fit_predict(points[:, 0:2])

    # Plot the centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='red', zorder=10)
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2])
    plt.title("K-Means")
    plt.show()


def run_kmedoids(points, n_clusters, use_labels="True", rng=0):
    """
    Run kmedoids algorithm on the given points with the given number of clusters and plot the centroids.

    :param points: The dataset.
    :param n_clusters: The number of clusters to be found in the data.
    :param use_labels: Whether to use the labels of the points as an attribute.
    :param rng: RandomState object, or seed 0 by default

    """
    # Set random initial medoids.
    rng = check_random_state(rng)
    initial_medoids = rng.randint(len(points), size=n_clusters)

    # Create instance of K-Medoids algorithm.
    if use_labels:
        kmedoids_instance = kmedoids(points, initial_medoids)
    else:
        kmedoids_instance = kmedoids(points[:, 0:2], initial_medoids)

    # Run cluster analysis and obtain results.
    kmedoids_instance.process()

    # Plot the centroids
    centroids = points[kmedoids_instance.get_medoids()]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='red', zorder=10)
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2])
    plt.title("K-Medoids")
    plt.show()
