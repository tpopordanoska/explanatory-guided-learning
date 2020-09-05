import pandas as pd
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment
from src.models.learners import *


class BloodTransfusion(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"]
        self.load_dataset('data', urls)

        columns = ['recency', 'frequency', 'cc', 'time', 'donated']
        dataset = pd.read_csv("data/transfusion.data", delimiter=',', skiprows=1, names=columns)

        # 178 1s, 569 0s
        y = dataset['donated'].to_numpy()
        X = dataset.drop('donated', axis=1)

        super().__init__(model, X, y, feature_names=X.columns, name="Blood Transfusion",
                         prop_known=0.001, rng=rng, normalizer=StandardScaler())
