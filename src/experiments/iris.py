import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from src.models.learners import GradientBoosting
from .experiment import Experiment


class Iris(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        dataset = load_iris()
        X = dataset.data
        # class 0: 50, class 1: 50, class 2: 50
        y = pd.Series(dataset.target).map({0: 0, 1: 0, 2: 1}).to_numpy()
        feature_names = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width']

        super().__init__(model, X, y, feature_names=feature_names, name="Iris", prop_known=0.01,
                         rng=rng, normalizer=StandardScaler())
