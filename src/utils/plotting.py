import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from src.experiments import Adult
from src.utils.normalizer import Normalizer

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


def plot_decision_surface(running_instance, query_idx=None, y_pred=None, soft=True, title=""):
    """
    Plots the decision surface of the model together with the data points.

    :param running_instance: An object containing the information about the current experiment and arguments
    :param query_idx: The index of chosen least confident example
    :param y_pred: The predictions of the model
    :param soft: Whether to plot  kernel-like boundary
    :param title: The title of the plot

    """
    experiment = running_instance.experiment
    known_idx = running_instance.known_idx
    train_idx = running_instance.train_idx

    if experiment.X.shape[1] > 2:
        plot_decision_surface_tsne(running_instance, query_idx, y_pred, title)
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

        save_plot(plt, experiment.path, title, title, False)


def plot_decision_surface_tsne(running_instance, query_idx, y_pred, title):
    exp = running_instance.experiment
    X_known, y_known, X_train, y_train, _, _ = running_instance.get_all_data()
    X_known_norm, X_train_norm = Normalizer(exp.normalizer).normalize_known_train(X_known, X_train)

    X_known_train_norm = np.concatenate((X_known_norm, X_train_norm), axis=0)

    X_initial = Normalizer(exp.normalizer).normalize(exp.X)
    X_embedded = get_tsne_embedding(X_initial)
    X_known_embedded = X_embedded[running_instance.known_idx]
    X_train_embedded = X_embedded[running_instance.train_idx]
    X_known_train_embedded = np.concatenate((X_known_embedded, X_train_embedded), axis=0)

    y_predicted = running_instance.predict(X_known_train_norm)

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
        least_conf = running_instance.get_from_indexes(exp.X, query_idx)
        if isinstance(exp, Adult):
            X_train = X_train.to_numpy()
        idx_array = np.where((X_train[:, 0] == least_conf[0]) & (X_train[:, 1] == least_conf[1]))[0]
        if len(idx_array):
            idx_in_train = idx_array[0]
            least_conf_norm = running_instance.get_from_indexes(X_train_embedded, idx_in_train)
            plt.scatter(least_conf_norm[0], least_conf_norm[1], marker='x', s=400, linewidths=5, color='yellow')

    save_plot(plt, exp.path, title, title, False)


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


def plot_rules(clf, X, y, title, path, rules_f1):

    if X.shape[1] > 2:
        plot_rules_tsne(X, y, title, path)
    else:
        figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
        xx, yy = create_meshgrid(X, 0.005)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, s=45)

        save_plot(plt, path, " rules " + str(title), "F1 rules wrt svm: {}".format(rules_f1), False)


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


def plot_initial_points(known_idx, test_idx, experiment):

    X_initial = Normalizer(experiment.normalizer).normalize(experiment.X)
    if experiment.X.shape[1] > 2:
        X_initial = get_tsne_embedding(X_initial)

    plot_points(X_initial, experiment.y, "Initial points", experiment.path)
    plot_points(X_initial[known_idx], experiment.y[known_idx], "Known points", experiment.path)
    plot_points(X_initial[test_idx], experiment.y[test_idx], "Test points", experiment.path)


def plot_points(X, y, title="", path=None):
    figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=45, cmap=plt.cm.coolwarm)

    save_plot(plt, path, title, title)


def get_tsne_embedding(X):
    return TSNE(n_components=2, n_iter=300, random_state=0).fit_transform(X)


def save_plot(plt, path, img_name, plot_title=None, use_grid=True, use_date=True):
    img_name = "{}.pdf".format(img_name)
    plt.grid(use_grid)
    if plot_title is not None:
        plt.title(plot_title)
    if path:
        try:
            if use_date:
                img_name = "{}-{}.pdf".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f'), img_name)
            plt.savefig(os.path.join(path, img_name), bbox_inches='tight', format='pdf')
        except ValueError:
            print("Something went wrong while saving image: ", img_name)
    else:
        plt.show()
    plt.close()
