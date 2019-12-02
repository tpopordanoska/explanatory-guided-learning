import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedKFold


class Experiment:
    def __init__(self, X, y, rng=None):
        self.X, self.y = X, y
        self.rng = check_random_state(rng)

    def split(self, n_splits=10, prop_known=0.5):

        # Generate folds
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.rng)
        for nontest_indices, test_indices in kfold.split(self.X, self.y):

            # Split the non-test set into known set and training set
            n_known = max(int(len(nontest_indices) * prop_known), 5)
            known_indices = self.rng.choice(nontest_indices, size=n_known)
            tr_indices = np.array(list(set(nontest_indices) - set(known_indices)))

            assert len(set(known_indices) & set(tr_indices)) == 0
            assert len(set(known_indices) & set(test_indices)) == 0
            assert len(set(tr_indices) & set(test_indices)) == 0

            yield known_indices, tr_indices, test_indices


