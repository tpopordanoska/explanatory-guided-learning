import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.learners import GradientBoosting
from .experiment import Experiment


class Glass(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"]
        self.load_dataset('data', urls)

        columns = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
        dataset = pd.read_csv("data/glass.data", delimiter=',', names=columns)

        # value counts: 1: 70, 2: 76, 3: 17, 5:13, 6:9, 7:29
        y = dataset['Type'].map({
            1: 0,
            2: 0,
            3: 1,
            5: 1,
            6: 1,
            7: 1
        }).to_numpy()
        X = dataset.drop('Type', axis=1)

        super().__init__(model, X, y, feature_names=columns, name="Glass", prop_known=0.01,
                         rng=rng, normalizer=StandardScaler())
