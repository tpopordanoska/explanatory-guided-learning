import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .experiment import Experiment
from ..learners import *


class Hepatitis(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = RandomForrest(n_estimators=5, max_depth=3)

        urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"]
        self.load_dataset('data', urls)

        columns = ["CLASS", "AGE", "SEX", "STEROID", "ANTIVIRAL", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER_BIG",
                   "LIVER_FIRM", "SPLEEN_PALPABLE",	"SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALKPHOSPHATE", "SGOT",
                   "ALBUMIN", "PROTIME", "HISTOLOGY"]
        dataset = pd.read_csv("data/hepatitis.data", delimiter=',', names=columns)

        # 123 0s and 23 1s
        y = dataset['CLASS'].map({1: 1, 2: 0}).to_numpy()
        X = dataset.drop('CLASS', axis=1)

        X = X.replace({'?': np.nan})
        X.fillna(X.median(), inplace=True)
        str_columns = X.select_dtypes(include=["object"])
        X[str_columns.columns] = str_columns.astype(float)

        super().__init__(model, X, y, feature_names=X.columns, name="Hepatitis", metric="f1",
                         prop_known=0.001, rng=rng, normalizer=MinMaxScaler())
