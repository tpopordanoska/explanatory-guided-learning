from src.running_instance import RunningInstance
import numpy as np


class GuidedLearning(RunningInstance):

    def __init__(self, **kwargs):
        self.query_array = []
        super().__init__(**kwargs)

    def query(self):
        if not len(self.query_array):
            # It has not been initialized
            queried = 0
            query_array = []
            train_idx = self.train_idx.copy()
            y_train = self.experiment.y[train_idx]
            # while we haven't emptied the training pool and we haven't reached the max_iter
            while len(y_train) and queried < self.args.max_iter:
                for label in np.unique(y_train):
                    sampled_subset_idx = np.where(y_train == label)[0]
                    if not len(sampled_subset_idx):
                        continue
                    idx_in_train = self.select_random(sampled_subset_idx, self.experiment.rng)
                    query_array.append(train_idx[idx_in_train])
                    train_idx = np.delete(train_idx, idx_in_train)
                    queried += 1
                    y_train = self.experiment.y[train_idx]
            self.query_array = query_array

        return self.query_array[self.iteration]
