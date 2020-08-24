import numpy as np
import pandas as pd
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


class Magic(Experiment):
    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        columns = [
            'fLength', 'fWidth', 'fSize',  'fConc',  'fConc1', 'fAsym',
            'fM3Long', 'fM3Trans', 'fAlpha', 'fDist',  'class',
        ]
        df = pd.read_csv(join('data', 'magic04.data'), names=columns, sep=',')

        _tocat(df, 'class')
        y = df['class'].to_numpy().astype(int)
        nony = list(sorted(set(df.columns) - {'class'}))
        X = df.loc[:, nony]

        sampled_idx, _ = list(StratifiedShuffleSplit(n_splits=2,
                                                     train_size=0.2,
                                                     random_state=0).split(X, y))[0]
        X, y = X.iloc[sampled_idx], y[sampled_idx]

        super().__init__(model, X, y,
                         feature_names=X.columns,
                         name=f"Magic 04",
                         prop_known=0.001,
                         normalizer=Normalizer(),
                         rng=rng)
