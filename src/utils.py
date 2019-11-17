import numpy as np
import matplotlib.pyplot as plt


def plot_decision_surface(model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, y_pred=None, least_conf=None, soft=True, title = ""):
    """
    Plots the decision surface of the model together with the data points.

    :param model: The trained model
    :param X_labeled: The labeled data points used for training
    :param y_labeled: The labels for the training data points
    :param X_unlabeled: The unlabeled data points
    :param y_unlabeled: The true labels for the "unlabeled"
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
        plt.scatter(least_conf[0], least_conf[1], marker='x', s=169, linewidths=8, color='white', zorder=10)
    plt.title(title)
    plt.show()


def plot_points(X, y, title=""):
    """
    Plot the given points with their corresponding labels.

    :param X: Contains the coordinates of the points to be plotted
    :param y: The corresponding labels
    :param title: The title of the plot

    """
    plt.scatter(X[:,0], X[:,1], c= y, marker='o', s=100, edgecolor='k', cmap=plt.cm.coolwarm)
    # set axes range
    plt.xlim(-0.2, 4.2)
    plt.ylim(-0.2, 4.2)
    plt.title(title)
    plt.show()


def plot_acc(acc_scores):
    """
    Plot the accuracy scores as a function of the queried instances.

    :param acc_scores: The accuracy scores to be plotted

    """
    x = np.arange(len(acc_scores))
    plt.plot(x, acc_scores)
    plt.grid(True)
    plt.xlabel('# instances queried')
    plt.ylabel('Accuracy score')
    plt.title('Accuracy as a function of # instances queried')
    plt.show()


def least_confident_idx(model, examples):
    """
    Get the index of the example closest to the decision boundary.

    :param model: The trained model
    :param examples: The data points

    :return: The index of the least confident example
    """
    margins = np.abs(model.decision_function(examples))
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


def select_random(data):
    """
    Get a random element from the given data.

    :param data: The data to find a random element from
    :return: A random element from the data
    """
    return data[np.random.choice(data.shape[0], replace=False)]
