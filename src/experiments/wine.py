import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from src.models.learners import GradientBoosting
from .experiment import Experiment


class Wine(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        dataset = load_wine()
        X = dataset.data

        # class 1: 59, class 2: 71, class 3: 48
        y = pd.Series(dataset.target).map({0: 0, 1: 0, 2: 1}).to_numpy()
        # Fix name of the feature causing issues with rules
        dataset.feature_names[11] = 'od280_od315_of_diluted_wines'

        super().__init__(model, X, y, feature_names=dataset.feature_names, name="Wine", prop_known=0.01,
                         rng=rng, normalizer=StandardScaler())
