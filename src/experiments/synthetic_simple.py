import numpy as np
from sklearn.datasets.samples_generator import make_blobs

from src.learners import *
from .experiment import Experiment


class SyntheticSimple(Experiment):
    def __init__(self, model, rng, balanced_db=True, tiny_clusters=True):

        rng = check_random_state(rng)
        # Generate mock data with balanced number of positive and negative examples
        X_pos, y_pos = self.generate_positive(6)
        X_neg, y_neg = self.generate_negative(7, 100, rng)
        if not balanced_db:
            # Generate mock data with rare grid class
            proportion = 0.3
            perm = rng.permutation(len(X_pos))
            X_pos = X_pos[perm[: int(proportion*len(X_pos))]]
            y_pos = y_pos[perm[: int(proportion*len(X_pos))]]

        if tiny_clusters:
            # Generate tiny clusters (0 to 5 points) around the positive points
            centers = X_pos
            n_samples = rng.randint(0, 5, size=len(centers))
            cluster_std = rng.uniform(0, 0.2, size=len(centers))

            X_pos_add, _ = make_blobs(n_samples=n_samples, cluster_std=cluster_std, centers=centers, n_features=2,
                                      random_state=1)
            X_pos = np.concatenate((X_pos, X_pos_add), axis=0)
            y_pos = np.ones((len(X_pos)), dtype=int)

        X = np.concatenate((X_pos, X_neg), axis=0)
        y = np.concatenate((y_pos, y_neg), axis=0)

        super().__init__(model, X, y, feature_names=['x', 'y'], name="Synthetic simple", prop_known=0.2, rng=rng)

    @staticmethod
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

    def generate_negative(self, axis, how_many, rng=0):
        """
        Generate randomly distributed points
        :param axis: Specifies the length of the axis
        :param how_many: How many points to be generated
        :return: Coordinates of the generated points and an array of zeros as their labels
        """
        pointsn = rng.rand(how_many, 2) * (axis - 1)
        yn = np.zeros((len(pointsn)), dtype=int)
        return pointsn, yn

    def generate_points(self, axis, how_many):
        """
        Generate points of class 1 that lie on a grid and points of class 0 that are randomly distributed
        :param axis: the length of the axis
        :param how_many: how many points of class 0 to be generated
        :return: coordinates of the generated points and an array with the corresponding labels
        """
        pointsp, yp = self.generate_positive(axis)
        pointsn, yn = self.generate_negative(axis, how_many)

        X = np.concatenate((pointsp, pointsn), axis=0)
        y = np.concatenate((yp, yn), axis=0)
        return X, y