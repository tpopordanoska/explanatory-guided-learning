import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state


def create_folders():
    path_results = os.getcwd() + "\\results"
    try:
        os.mkdir(path_results)
    except FileExistsError:
        print("Directory ", path_results, " already exists")
    except OSError:
        print("Creation of the directory %s failed" % path_results)
    else:
        print("Successfully created the directory %s " % path_results)

    # Create a separate folder for each time running the experiment
    path = path_results + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory ", path, " already exists")
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    return path


def plot_decision_surface(model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, y_pred=None,
                          least_conf=None, soft=True, title = "", path=None):
    """
    Plots the decision surface of the model together with the data points.

    :param model: The trained model
    :param X_labeled: The labeled data points used for training
    :param y_labeled: The labels for the training data points
    :param X_unlabeled: The unlabeled data points
    :param y_unlabeled: The true labels for the "unlabeled"
    :param y_pred: The predictions of the model
    :param least_conf: The chosen least confident example
    :param soft: Whether to plot  kernel-like boundary

    """
    # create a mesh to plot in
    h = .02  # step size in the mesh
    X = np.concatenate((X_labeled, X_unlabeled), axis=0)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if (soft):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else :
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    if y_pred is not None:
        plt.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c=y_pred, cmap=plt.cm.coolwarm)
    else:
        plt.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c=y_unlabeled, cmap=plt.cm.coolwarm)
        plt.scatter(X_labeled[:, 0], X_labeled[:, 1], c=y_labeled, cmap=plt.cm.coolwarm, linewidths=6)

    if least_conf is not None:
        plt.scatter(least_conf[0], least_conf[1], marker='x', s=169, linewidths=8, color='green', zorder=10)
    plt.title(title)
    if path:
        plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "-" + title + '.png')
    else:
        plt.show()
    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.close()


def plot_points(X, y, title="", path=None):
    """
    Plot the given points with their corresponding labels.

    :param X: Contains the coordinates of the points to be plotted
    :param y: The corresponding labels
    :param title: The title of the plot

    """
    plt.scatter(X[:,0], X[:,1], c= y, marker='o', s=30, edgecolor='k', cmap=plt.cm.coolwarm)
    # set axes range
    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.title(title)
    if path:
        plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "-" + title + '.png')
    else:
        plt.show()
    plt.close()


def plot_acc(scores, stds, labels, title="", path=None):
    """
    Plot the accuracy scores as a function of the queried instances.

    :param scores: The accuracy scores to be plotted
    :param labels: The labels to be displayed in the legend for the accuracy scores

    """
    # Filter out empty arrays or nans
    filtered_scores = [score for score in scores if not np.isnan(score).all()]
    # TODO: check if they are of equal size
    for score, std, label in zip(filtered_scores, stds, labels):
        x = np.arange(len(score))
        plt.plot(x, score, label=label)
        plt.fill_between(x, score - std, score + std, alpha=0.15, linewidth=0)
    plt.grid(True)
    plt.xlabel('# instances queried')
    plt.ylabel('F1 score')
    plt.title(title)
    plt.legend()
    if path:
        plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "-" + title + '.png')
    else:
        plt.show()
    plt.close()


def least_confident_idx(model, examples):
    """
    Get the index of the example closest to the decision boundary.

    :param model: The trained model
    :param examples: The data points

    :return: The index of the least confident example
    """
    margins = np.abs(model.decision_function(examples))
    if len(margins) == 0:
        return None
    return np.argmin(margins)


def select_by_coordinates(x, y, data):
    """
    Get the index of the element in the data, if found

    :param x: The x coordinate
    :param y: The y coordinate
    :param data: The data to find the element by coordinates from
    :return: The index of the found element
    """
    # TODO: take care of element not found and fix [][]
    return [np.where((data[:, 0] == x) & (data[:, 1] == y))[0][0]]


def select_random(data, rng=0):
    """
    Get a random element from the given data.

    :param data: The data to find a random element from
    :param rng: RandomState object, or seed 0 by default

    :return: A random element from the data
    """
    rng = check_random_state(rng)
    return data[rng.choice(data.shape[0], replace=False)]


def concatenate_data(X_train, y_train, X_unlabeled, y_unlabeled, y_pred):
    """
    Concatenate the given data in one matrix with three columns.

    :param X_train: The coordinates of the points of the first data matrix
    :param y_train: The labels of the points from the first data matrix
    :param X_unlabeled: The coordinates of the points of the second data matrix
    :param y_unlabeled: The labels of the points from the second data matrix
    :param y_pred: The predictions of the unlabeled points

    :return: One matrix with three columns (coordinates and label) of the concatenated data
    """
    Xy_train = np.concatenate((X_train, np.array([y_train]).T), axis=1)
    Xy_unlabeled = np.concatenate((X_unlabeled, np.array([y_unlabeled]).T), axis=1)
    Xy = np.concatenate((Xy_train, Xy_unlabeled), axis=0)
    labels = np.concatenate((y_train, y_pred))
    return np.concatenate((Xy, np.array([labels]).T), axis=1)