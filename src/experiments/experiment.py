import os

import numpy as np
import requests
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state


class Experiment:
    """
    Class containing common methods for the experiments
    """
    def __init__(self, model, X, y, feature_names, name, prop_known=0.1, metric="f1", rng=None):
        self.model = model
        self.X, self.y = X, y
        self.feature_names = feature_names
        self.name = name
        self.metric = metric
        self.prop_known = prop_known
        self.rng = check_random_state(rng)

    @staticmethod
    def load_dataset(path, urls):
        """
        Download the content for a list of URLs and save them to a folder

        :param path: The path to the location where the data will be saved
        :param urls: The list of URLs from which the content will be downloaded and saved

        """
        if not os.path.exists(path):
            os.mkdir(path)

        for url in urls:
            data = requests.get(url).content
            filename = os.path.join(path, os.path.basename(url))
            with open(filename, "wb") as file:
                file.write(data)

    def split(self, n_splits=10, prop_known=0.5):
        """
        Split the data into known, train and test set.

        :param n_splits: How many splits to generate
        :param prop_known: The proportion of the known points

        :return: Three arrays of indices for known, train and test set respectfully
        """
        # Generate folds
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.rng)
        for nontest_indices, test_indices in kfold.split(self.X, self.y):

            # Split the non-test set into known set and training set
            n_known = max(int(len(nontest_indices) * prop_known), 5)
            known_indices = self.rng.choice(nontest_indices, size=n_known)
            # Check if each class has at least 2 examples and if not, generate new random indices
            _, counts = np.unique(self.y[known_indices], return_counts=True)
            num_classes = len(np.unique(self.y))
            if not (len(counts) == num_classes and all(i > 1 for i in counts)):
                new_seed = 1
                while not (len(counts) == num_classes and all(i > 1 for i in counts)):
                    np.random.seed(new_seed)
                    new_seed += 1
                    known_indices = np.random.choice(nontest_indices, size=n_known)
                    _, counts = np.unique(self.y[known_indices], return_counts=True)
            tr_indices = np.array(list(set(nontest_indices) - set(known_indices)))

            assert len(set(known_indices) & set(tr_indices)) == 0
            assert len(set(known_indices) & set(test_indices)) == 0
            assert len(set(tr_indices) & set(test_indices)) == 0

            yield known_indices, tr_indices, test_indices
