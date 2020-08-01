import os

import numpy as np
import sklearn.metrics.pairwise as metrics
from tqdm import tqdm

from src.running_instance import RunningInstance
from src.utils.normalizer import Normalizer


class DensityWeightedAL(RunningInstance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self):
        X_known, y_known, X_train, y_train, _, _ = self.get_all_data()
        X_known_norm, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)
        margins = self.get_margins(X_train_norm)

        # Load or compute the similarity matrix for all data
        dist_matrix_path = os.path.join(os.getcwd(), "distance_matrix")
        self.load_similarity_matrix_if_exists(dist_matrix_path)
        if self.cos_distance_matrix is None:
            self.compute_and_save_similarity_matrix(dist_matrix_path)

        cos_distance = self.get_similarity_matrix_for_train_data()
        query_idx = np.argmin(margins * cos_distance ** self.param)

        return self.train_idx[query_idx]

    def load_similarity_matrix_if_exists(self, path):
        try:
            self.cos_distance_matrix = np.load(os.path.join(path, self.experiment.name + ".npy"))
        except IOError:
            print("File {} does not exist or cannot be read".format(path))

    def compute_and_save_similarity_matrix(self, path):
        """
        Compute the similarity matrix if it has not been stored already

        :param path: The path where the similarity matrix will be saved
        """
        X_norm = Normalizer(self.experiment.normalizer).normalize(self.experiment.X)
        self.cos_distance_matrix = np.zeros((len(X_norm), len(X_norm)))
        for i, x_i in tqdm(enumerate(X_norm)):
            for j, x_j in enumerate(X_norm):
                self.cos_distance_matrix[i][j] = metrics.cosine_distances(
                    x_i.reshape(1, -1), x_j.reshape(1, -1))[0][0]
        try:
            os.mkdir(path)
        except FileExistsError:
            print("Directory {} already exists".format(path))

        np.save(os.path.join(path, self.experiment.name), self.cos_distance_matrix)

    def get_similarity_matrix_for_train_data(self):
        cos_distance = []
        for idx in self.train_idx:
            dists = self.cos_distance_matrix[idx, [self.train_idx]]
            cos_distance.append(np.mean(dists))

        return np.asarray(cos_distance)
