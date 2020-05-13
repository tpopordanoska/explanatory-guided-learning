import pandas as pd
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment


class Hepatitis(Experiment):

    def __init__(self, **kwargs):
        model = kwargs.pop("model")

        urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"]
        self.load_dataset('data', urls)

        dataset = pd.read_csv("data/hepatitis.data", delimiter=' ', skiprows=1)

        y = dataset['CLASS'].to_numpy()
        X = dataset.drop('CLASS', axis=1)

        super().__init__(model, X, y, feature_names=X.columns, name="Hepatitis", metric="f1",
                         prop_known=0.001, rng=model.rng, normalizer=StandardScaler())

