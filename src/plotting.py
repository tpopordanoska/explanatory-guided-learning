import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from .normalizer import *
from .utils import *

sns.set()
sns.set_context("paper")


def create_meshgrid(points, h=0.1):
    """
    Create the mesh grid to plot in.

    :param h: The mesh size
    :param points: The dataset

    :return: The mesh grid to plot in
    """
    x_min, x_max = points[:, 0].min() - 0.1, points[:, 0].max() + 0.1
    y_min, y_max = points[:, 1].min() - 0.1, points[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return xx, yy


def plot_decision_surface_tsne(experiment, known_idx, train_idx, query_idx, y_pred, title, path):
    """

    Plots the decision surface of the model on TSNE-embedded data.

    :param experiment: The experiment
    :param known_idx: The indexes of the known points
    :param train_idx: The indexes of the training points
    :param query_idx: The index of chosen least confident example
    :param y_pred: The predictions of the model
    :param title: The title of the plot
    :param path: The path of the folder where the plot will be saved

    """
    X_known, y_known = get_from_indexes(experiment.X, known_idx), experiment.y[known_idx]
    X_train, y_train = get_from_indexes(experiment.X, train_idx), experiment.y[train_idx]
    X_known_norm, X_train_norm = Normalizer(experiment.normalizer).normalize_known_train(X_known, X_train)
    X_known_train_norm = np.concatenate((X_known_norm, X_train_norm), axis=0)

    X_initial = Normalizer(experiment.normalizer).normalize(experiment.X)
    X_embedded = get_tsne_embedding(X_initial)
    X_known_embedded = X_embedded[known_idx]
    X_train_embedded = X_embedded[train_idx]
    X_known_train_embedded = np.concatenate((X_known_embedded, X_train_embedded), axis=0)

    y_predicted = experiment.model.predict(X_known_train_norm)

    h = 10
    x_min, x_max = X_known_train_embedded[:, 0].min() - 1, X_known_train_embedded[:, 0].max() + 1
    y_min, y_max = X_known_train_embedded[:, 1].min() - 1, X_known_train_embedded[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_known_train_embedded, y_predicted)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((h, h))

    plt.figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
    plt.contourf(xx, yy, voronoiBackground, cmap=plt.cm.RdBu_r, alpha=0.8)

    if y_pred is not None:
        plt.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], c=y_pred, cmap=plt.cm.RdBu_r, s=45)
    else:
        plt.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], c=y_train, cmap=plt.cm.RdBu_r, s=45)
        plt.scatter(X_known_embedded[:, 0], X_known_embedded[:, 1], c=y_known, cmap=plt.cm.RdBu_r, s=45,
                    edgecolors="yellow", linewidths=2)

    if query_idx is not None:
        least_conf = get_from_indexes(experiment.X, query_idx)
        if isinstance(experiment, Adult):
            X_train = X_train.to_numpy()
        idx_array = np.where((X_train[:, 0] == least_conf[0]) & (X_train[:, 1] == least_conf[1]))[0]
        if len(idx_array):
            idx_in_train = idx_array[0]
            least_conf_norm = get_from_indexes(X_train_embedded, idx_in_train)
            plt.scatter(least_conf_norm[0], least_conf_norm[1], marker='x', s=400, linewidths=5, color='yellow')

    save_plot(plt, path, title, title, False)


def plot_decision_surface(experiment, known_idx, train_idx, query_idx=None, y_pred=None, soft=True, title="",
                          path=None):
    """
    Plots the decision surface of the model together with the data points.

    :param experiment: The experiment
    :param known_idx: The indexes of the known points
    :param train_idx: The indexes of the training points
    :param query_idx: The index of chosen least confident example
    :param y_pred: The predictions of the model
    :param soft: Whether to plot  kernel-like boundary
    :param title: The title of the plot
    :param path: The path of the folder where the plot will be saved

    """
    if experiment.X.shape[1] > 2:
        plot_decision_surface_tsne(experiment, known_idx, train_idx, query_idx, y_pred, title, path)
    else:
        X_known, y_known = experiment.X[known_idx], experiment.y[known_idx]
        X_train, y_train = experiment.X[train_idx], experiment.y[train_idx]
        X_known_norm, X_train_norm = Normalizer(experiment.normalizer).normalize_known_train(X_known, X_train)
        X_known_train_norm = np.concatenate((X_known_norm, X_train_norm), axis=0)
        model = experiment.model

        figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
        # create a mesh to plot in
        h = 0.05  # step size in the mesh
        xx, yy = create_meshgrid(X_known_train_norm, h)

        if soft:
            if hasattr(model.sklearn_model, "decision_function"):
                Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.8)

        if y_pred is not None:
            wrong_points = X_train_norm[np.where(y_pred != y_train)]
            plt.scatter(X_train_norm[:, 0], X_train_norm[:, 1], c=y_pred, cmap=plt.cm.RdBu_r, s=45)
            plt.scatter(wrong_points[:, 0], wrong_points[:, 1], s=120, cmap=plt.cm.RdBu_r, facecolors='none',
                        edgecolors='green', linewidths=2)
        else:
            plt.scatter(X_train_norm[:, 0], X_train_norm[:, 1], c=y_train, cmap=plt.cm.RdBu_r, s=45)
            plt.scatter(X_known_norm[:, 0], X_known_norm[:, 1], c=y_known, cmap=plt.cm.RdBu_r, s=45,
                    edgecolors="yellow", linewidths=2)

        if query_idx is not None:
            least_conf = experiment.X[query_idx]
            idx_array = np.where((X_train[:, 0] == least_conf[0]) & (X_train[:, 1] == least_conf[1]))[0]
            if len(idx_array):
                idx_in_train = idx_array[0]
                least_conf_norm = X_train_norm[idx_in_train]
                plt.scatter(least_conf_norm[0], least_conf_norm[1], marker='x', s=400, linewidths=5, color='yellow')

        plt.xlim(X_known_train_norm[:, 0].min() - 0.1, X_known_train_norm[:, 0].max() + 0.1)
        plt.ylim(X_known_train_norm[:, 1].min() - 0.1, X_known_train_norm[:, 1].max() + 0.1)

        save_plot(plt, path, title, title, False)


def plot_kmedoids(points, kmedoids_instance, other_points, path):
    # Plot the decision boundary
    plt.figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
    plt.xlim(points[:, 0].min() - 0.1, points[:, 0].max() + 0.1)
    plt.ylim(points[:, 1].min() - 0.1, points[:, 1].max() + 0.1)
    xx, yy = create_meshgrid(points)

    # Obtain labels for each point in mesh.
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
    centroids_idx = kmedoids_instance.get_medoids()
    centroids = points[centroids_idx]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='red', zorder=10)

    save_plot(plt, path, "K-Medoids", "K-Medoids")

def plot_rules_tsne(X, y, title, path):

    X_embedded = get_tsne_embedding(X)
    h = 10
    x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
    y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_embedded, y)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((h, h))

    plt.figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
    plt.contourf(xx, yy, voronoiBackground, cmap=plt.cm.RdBu_r, alpha=0.8)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.RdBu_r, s=45)

    save_plot(plt, path, title, " rules", False)


def plot_rules(clf, X, y, title, path, rules_f1):

    if X.shape[1] > 2:
        plot_rules_tsne(X, y, title, path)
    else:
        figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
        xx, yy = create_meshgrid(X, 0.005)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, s=45)

        save_plot(plt, path, " rules " + str(title), "F1 rules wrt svm: {}".format(rules_f1), False)


def plot_points(X, y, title="", path=None):
    """
    Plot the given points with their corresponding labels.

    :param X: Contains the coordinates of the points to be plotted
    :param y: The corresponding labels
    :param title: The title of the plot
    :param path: The path of the folder where the plot will be saved

    """
    figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=45, cmap=plt.cm.coolwarm)

    save_plot(plt, path, title, title)


def get_tsne_embedding(X):
    return TSNE(n_components=2, n_iter=300, random_state=0).fit_transform(X)
