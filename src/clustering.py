from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import KMeans


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


def create_meshgrid(points):
    """
    Create the mesh grid to plot in.

    :param points: The dataset

    :return: The mesh grid to plot in
    """
    # Plot the decision boundary
    h = 0.1
    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return xx, yy


def run_kmedoids(points_pd, n_clusters, other_points=None, use_labels="False", use_weights="False", path=None, plots_off=True):
    """
    Run kmedoids algorithm on the given points with the given number of clusters and plot the centroids.

    :param points_pd: Pandas DataFrame containing the features and the predicted labels on the train set
    :param n_clusters: The number of clusters to be found in the data
    :param other_points: Other points (ex. wrongly classified) to plot instead of the points used to find clusters
    :param use_labels: Whether to use the labels of the points as an attribute
    :param use_weights: Whether to weight the labels by the number of other attributes
    :param path: The path to the folder where the graphs will be saved
    :param plots_off: Whether to plot the graphs

    :return: Indexes (in points) of the found clusters and the corresponding centroids
    """
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

    np.random.seed(0)
    # 1. Set random initial medoids.
    initial_medoids = np.random.randint(len(points), size=n_clusters)

    # 2. Create instance of K-Medoids algorithm.
    kmedoids_instance = kmedoids(points, initial_medoids)
    # Run cluster analysis and obtain results.
    kmedoids_instance.process()

    # 3. Get the centroids
    centroids_idx = kmedoids_instance.get_medoids()
    centroids = points[centroids_idx]

    if not plots_off:
        # Plot the decision boundary
        plt.figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        xx, yy = create_meshgrid(points)

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmedoids_instance.predict((np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        if other_points is not None:
            plt.scatter(other_points.iloc[:, 0], other_points.iloc[:, 1], c=other_points.predictions, marker='o', s=30)
        else:
            plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], marker='o', s=30)
        # Plot the centroids

        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='red', zorder=10)
        plt.title("K-Medoids")
        if path:
            plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + " K-medoids.png")
        else:
            plt.show()
        plt.close()
    return kmedoids_instance.get_clusters(), centroids_idx

