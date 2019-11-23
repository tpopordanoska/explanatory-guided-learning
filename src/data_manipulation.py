import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedShuffleSplit


def generate_positive(axis):
    """
    Generate points that lie on a grid

    :param axis: How many numbers to be generated on each axis

    :return: Coordinates of the generated points and an array of ones as their labels
    """
    x_coord = np.arange(axis)
    y_coord = np.arange(axis)
    pointsp = np.array([[a, b] for a in x_coord for b in y_coord])
    yp = np.ones((len(pointsp)), dtype=int)
    return pointsp, yp


def generate_negative(axis, how_many, rng):
    """
    Generate randomly distributed points

    :param axis: Specifies the length of the axis
    :param how_many: How many points to be generated
    :param rng: RandomState object, or seed 0 by default

    :return: Coordinates of the generated points and an array of zeros as their labels
    """
    rng = check_random_state(rng)
    pointsn = rng.rand(how_many, 2) * (axis - 1)
    yn = np.zeros((len(pointsn)), dtype=int)
    return pointsn, yn


def generate_points(axis, how_many):
    """
    Generate points of class 1 that lie on a grid and points of class 0 that are randomly distributed
    :param axis: the length of the axis
    :param how_many: how many points of class 0 to be generated

    :return: coordinates of the generated points and an array with the corresponding labels
    """
    pointsp, yp = generate_positive(axis)
    pointsn, yn = generate_negative(axis, how_many)

    X = np.concatenate((pointsp, pointsn), axis=0)
    y = np.concatenate((yp, yn), axis=0)
    return X, y


def split_data(X, y, proportion, rng, n_splits=1):
    """
    Get indices to split the data in the given proportion,
    perserving the percentage of samples per class

    :param X: The features.
    :param y: The labels.
    :param proportion: The proportion of the dataset for the test split (second array).
    :param n_splits: The number of splitting iterations.

    :return: Two arrays of indices: first array is "train", second array is "test"
    """
    fold = StratifiedShuffleSplit(n_splits=n_splits, test_size=proportion, random_state=rng).split(X, y)
    return list(fold)[0]


def concatenate_data(X1, y1, X2, y2):
    """
    Concatenate the given data in one matrix with three columns.

    :param X1: The coordinates of the points of the first data matrix
    :param y1: The labels of the points from the first data matrix
    :param X2: The coordinates of the points of the second data matrix
    :param y2: The labels of the points from the second data matrix

    :return: One matrix with three columns (coordinates and label) of the concatenated data
    """
    Xy1 = np.concatenate((X1, np.array([y1]).T), axis=1)
    Xy2 = np.concatenate((X2, np.array([y2]).T), axis=1)
    return np.concatenate((Xy1, Xy2), axis=0)
