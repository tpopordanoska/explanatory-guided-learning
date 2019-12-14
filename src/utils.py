import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from .experiments import *
from .normalizer import *


def create_folders():
    """
    Create folders for storing the plots and the graphs.

    :return: The path to the created folder
    """
    path_results = "{}\\results".format(os.getcwd())
    try:
        os.mkdir(path_results)
    except FileExistsError:
        print("Directory {} already exists".format(path_results))
    except OSError:
        print("Creation of the directory {} failed".format(path_results))
    else:
        print("Successfully created the directory {} ".format(path_results))

    # Create a separate folder for each time running the experiment
    path = "{}\\{}". format(path_results, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f'))
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory ", path, " already exists")
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    return path


def create_experiment_folder(path, experiment, model):
    """
    Create folder for storing results for the given experiment and model

    :param path: Path to the folder created when the script is run
    :param experiment: The name of the experiment being performed
    :return: model: The name of the model currently running
    """
    path_experiment = "{}\\{}".format(path, experiment)
    try:
        os.mkdir(path_experiment)
    except OSError:
        print("Creation of the directory %s failed" % path_experiment)

    path_model = "{}\\{} {}".format(path_experiment, model, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.mkdir(path_model)
    except OSError:
        print("Creation of the directory %s failed" % path_model)

    return path_model


def plot_decision_surface(experiment, known_idx, train_idx, query_idx=None, y_pred=None, soft=True, title="", path=None):
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
    X_known, y_known = experiment.X[known_idx], experiment.y[known_idx]
    X_train, y_train = experiment.X[train_idx], experiment.y[train_idx]
    X_known_norm, X_train_norm = Normalizer(experiment.normalizer).normalize_known_train(X_known, X_train)
    model = experiment.model

    figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
    # create a mesh to plot in
    h = 0.05  # step size in the mesh

    x_min, x_max = X_known_norm[:, 0].min() - 1, X_known_norm[:, 0].max() + 1
    y_min, y_max = X_known_norm[:, 1].min() - 1, X_known_norm[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if soft:
        if hasattr(model._model , "decision_function"):
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    else :
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    if y_pred is not None:
        plt.scatter(X_train_norm[:, 0], X_train_norm[:, 1], c=y_pred, cmap=plt.cm.coolwarm, s=25)
    else:
        plt.scatter(X_train_norm[:, 0], X_train_norm[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=25)
        plt.scatter(X_known_norm[:, 0], X_known_norm[:, 1], c=y_known, cmap=plt.cm.coolwarm, s=25, linewidths=6)

    if query_idx is not None:
        least_conf = experiment.X[query_idx]
        idx_array = np.where((X_train[:, 0] == least_conf[0]) & (X_train[:, 1] == least_conf[1]))[0]
        if len(idx_array):
            idx_in_train = idx_array[0]
            least_conf_norm = X_train_norm[idx_in_train]
            plt.scatter(least_conf_norm[0], least_conf_norm[1], marker='x', s=500, linewidths=3, color='green')
    plt.title(title)
    # plt.xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    # plt.ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    if path:
        plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + "-" + title + '.png')
    else:
        plt.show()
    plt.close()


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
    # set axes range
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    # plt.xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    # plt.ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    plt.title(title)
    if path:
        plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + "-" + title + '.png')
    else:
        plt.show()
    plt.close()


def plot_acc(scores, stds, f1_score_passive, title="", path=None):
    """
    Plot the accuracy scores as a function of the queried instances.

    :param scores: Dictionary containing the accuracy scores for each method
    :param stds: Dictionary containing the standard deviations for each method
    :param f1_score_passive: The f1 score of the experiment in a passive setting
    :param title: The title of the plot
    :param path: The path of the folder where the plot will be saved

    """
    for key, score in scores.items():
        x = np.arange(len(score))
        plt.plot(x, score, label=key)
        plt.fill_between(x, score - stds[key], score + stds[key], alpha=0.25, linewidth=0)

    x = np.arange(len(max(scores.values(), key=lambda value: len(value))))
    passive_mean = np.array([f1_score_passive.mean() for i in range(len(x))])
    passive_std = np.array([f1_score_passive.std() * 2 for i in range(len(x))])

    plt.plot(x, passive_mean, label="Passive setting")
    plt.fill_between(x, passive_mean - passive_std, passive_mean + passive_std, alpha=0.25, linewidth=0)

    plt.grid(True)
    plt.xlabel('# instances queried')
    plt.ylabel('F1 score')
    plt.title(title)
    plt.legend()
    if path:
        plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + "-" + title + '.png')
    else:
        plt.show()
    plt.close()


def least_confident_idx(**kwargs):
    """
    Get the index of the example closest to the decision boundary.

    :param kwargs: Keyword arguments

    :return: The index (in X) of the least confident example
    """
    experiment = kwargs.pop("experiment")
    known_idx = kwargs.pop("known_idx")
    train_idx = kwargs.pop("train_idx")
    X_known = get_from_indexes(experiment.X, known_idx)
    X_train = get_from_indexes(experiment.X, train_idx)

    _, X_train_norm = Normalizer(experiment.normalizer).normalize_known_train(X_known, X_train)

    if hasattr(experiment.model._model, "decision_function"):
        margins = np.abs(experiment.model.decision_function(X_train_norm))
    elif hasattr(experiment.model._model, "predict_proba"):
        probs = experiment.model.predict_proba(X_train_norm)
        margins = np.sum(probs * np.log(probs), axis=1).ravel()
    else:
        raise AttributeError("Model with either decision_function or predict_proba method")

    if len(margins) == 0:
        return None
    return train_idx[np.argmin(margins)]


def get_from_indexes(X, indexes):
    if isinstance(X, pd.DataFrame):
        return X.iloc[indexes]
    return X[indexes]

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


def select_random(data, rng):
    """
    Get a random element from the given data.

    :param data: The data to find a random element from
    :param rng: RandomState object

    :return: A random element from the data
    """
    return rng.choice(data)


def concatenate_data(X_train, y_train, X_unlabeled, y_unlabeled, y_pred):
    """
    Concatenate the given data in one matrix with three columns.

    :param X_train: The coordinates of the points of the first data matrix
    :param y_train: The labels of the points from the first data matrix
    :param X_unlabeled: The coordinates of the points of the second data matrix
    :param y_unlabeled: The labels of the points from the second data matrix
    :param y_pred: The predictions of the unlabeled points

    :return: One matrix containing the features, the predictions and the true labels
    """
    Xy_train = np.concatenate((X_train, np.array([y_train]).T), axis=1)
    Xy_unlabeled = np.concatenate((X_unlabeled, np.array([y_pred]).T), axis=1)
    Xy = np.concatenate((Xy_train, Xy_unlabeled), axis=0)
    true_labels = np.concatenate((y_train, y_unlabeled))
    return np.concatenate((Xy, np.array([true_labels]).T), axis=1)