from src.running_instance import RunningInstance
import numpy as np
from src.utils.normalizer import Normalizer


class LeastConfidentAL(RunningInstance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self):
        X_known, X_train = self.get_known_train_features()

        _, X_train_norm = Normalizer(self.experiment.normalizer).normalize_known_train(X_known, X_train)
        margins = self.get_margins(X_train_norm)

        if len(margins) == 0:
            return None
        return self.train_idx[np.argmin(margins)]
