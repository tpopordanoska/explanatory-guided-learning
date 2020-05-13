import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .experiment import Experiment


class Heart(Experiment):

    def __init__(self, **kwargs):
        model = kwargs.pop("model")

        url = ["http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"]
        self.load_dataset('data', url)

        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                   'oldpeak', 'slope', 'ca', 'thal', 'target']

        dataset = pd.read_csv("data/processed.cleveland.data", delimiter=',', names=columns)

        # 123 0s and 23 1s
        y = dataset['CLASS'].map({1: 1, 2: 0}).to_numpy()
        X = dataset.drop('CLASS', axis=1)

        X = X.replace({'?': np.nan})
        X.fillna(X.median(), inplace=True)
        str_columns = X.select_dtypes(include=["object"])
        X[str_columns.columns] = str_columns.astype(float)

        super().__init__(model, X, y, feature_names=X.columns, name="Hepatitis", metric="f1",
                         prop_known=0.001, rng=model.rng, normalizer=MinMaxScaler())

