from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
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

def create_meshgrid(points, kmedoids_instance):
    # Plot the decision boundary
    h = .02
    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmedoids_instance.predict((np.c_[xx.ravel(), yy.ravel()]))
    # Put the result into a color plot
    return Z.reshape(xx.shape), xx, yy


def run_kmedoids(points, n_clusters, other_points=None, use_labels="False", rng=0, path=None, method=""):
    """
    Run kmedoids algorithm on the given points with the given number of clusters and plot the centroids.

    :param points: The dataset, including the predicted labels on the train and the real labels on the known set.
    :param n_clusters: The number of clusters to be found in the data.
    :param use_labels: Whether to use the labels of the points as an attribute.
    :param rng: RandomState object, or seed 0 by default

    """
    # Set random initial medoids.
    rng = check_random_state(rng)
    np.random.seed(0)
    initial_medoids = np.random.randint(len(points), size=n_clusters)

    # Create instance of K-Medoids algorithm.
    # if use_labels:
    #     kmedoids_instance = kmedoids(points, initial_medoids)
    # else:
    kmedoids_instance = kmedoids(points, initial_medoids)

    # Run cluster analysis and obtain results.
    kmedoids_instance.process()

    centroids_idx = kmedoids_instance.get_medoids()
    centroids = points[centroids_idx]

    # If the problem is two dimensional (the third column are the labels) then plot it
    if points.shape[1] == 3:
        # Plot the decision boundary
        Z, xx, yy = create_meshgrid(points, kmedoids_instance)

        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        if other_points is not None:
            plt.scatter(other_points.x, other_points.y, c=other_points.predictions, marker='o', s=30, edgecolor='k')
        else:
            plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], marker='o', s=30, edgecolor='k')
        # Plot the centroids

        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='red', zorder=10)
        plt.title("K-Medoids")
        if path:
            plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + " K-medoids " + method + ".png")
        else:
            plt.show()
        plt.close()
    return kmedoids_instance.get_clusters(), centroids_idx

