import numpy as np
import pandas as pd
import time, datetime
from os.path import join
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Normalizer

from .experiment import Experiment
from src.models.learners import *


def _tocat(df, column, value_to_i=None):
    if value_to_i is None:
        values = df[column].unique()
        value_to_i = {v: i for i, v in enumerate(list(sorted(values)))}
    df[column] = df[column].map(value_to_i)
    return value_to_i

def _totimestamp(s):
    return time.mktime(datetime.datetime.strptime(str(s), '%Y%m%d').timetuple())


class Risk(Experiment):
    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        df = pd.read_csv(join('data', 'RiskData.csv'), sep=',', na_values='.')
        df.dropna(inplace=True)

        _tocat(df, 'Region')
        _tocat(df, 'Bank_Relationship')

        df['Application_Date'] = df['Application_Date'].map(_totimestamp)
        df['Application_Date'] -= df['Application_Date'].min()

        y = df['Risk_Flag'].to_numpy().astype(int)
        nony = list(sorted(set(df.columns) - {'Risk_Flag'}))
        X = df.loc[:, nony]

        #sampled_idx, _ = list(StratifiedShuffleSplit(n_splits=2, train_size=0.1, random_state=0).split(X, y))[0]
        #X, y = X.iloc[sampled_idx], y[sampled_idx]

        super().__init__(model, X, y,
                         feature_names=X.columns,
                         name=f"Risk 04",
                         prop_known=0.001,
                         normalizer=Normalizer(),
                         rng=rng)
