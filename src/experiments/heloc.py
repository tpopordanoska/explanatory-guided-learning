import numpy as np
import pandas as pd
from os.path import join
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Normalizer

from .experiment import Experiment
from src.models.learners import *


class Heloc(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = SVM()

        df = pd.read_csv(join('data', 'heloc_dataset_v1.csv'), sep=',')

        y = df['RiskPerformance'].map({'Good': 1, 'Bad': 0}).to_numpy().astype(int)

        to_onehot = {
            # 0	derogatory comment
            0: (0, 0, 0, 0, 0, 0, 0, 0, 1),
            # 1	120+ days delinquent
            1: (0, 0, 0, 0, 0, 0, 0, 1, 0),
            # 2	90 days delinquent
            2: (0, 0, 0, 0, 0, 0, 1, 0, 0),
            # 3	60 days delinquent
            3: (0, 0, 0, 0, 0, 1, 0, 0, 0),
            # 4	30 days delinquent
            4: (0, 0, 0, 0, 1, 0, 0, 0, 0),
            # 5, 6	unknown delinquency
            5: (0, 0, 0, 1, 0, 0, 0, 0, 0),
            6: (0, 0, 0, 1, 0, 0, 0, 0, 0),
            # 7	current and never delinquent
            7: (0, 0, 1, 0, 0, 0, 0, 0, 0),
            # 8, 9	all other
            8: (0, 1, 0, 0, 0, 0, 0, 0, 0),
            9: (0, 1, 0, 0, 0, 0, 0, 0, 0),
            # XXX NOT DOCUMENTED
            -9: (1, 0, 0, 0, 0, 0, 0, 0, 0),
        }
        X1 = np.array([to_onehot[x] for x in list(df['MaxDelq2PublicRecLast12M'])])

        to_onehot = {
            # 1	No such value
            # 2	derogatory comment
            2: (0, 0, 0, 0, 0, 0, 0, 0, 1),
            # 3	120+ days delinquent
            3: (0, 0, 0, 0, 0, 0, 0, 1, 0),
            # 4	90 days delinquent
            4: (0, 0, 0, 0, 0, 0, 1, 0, 0),
            # 5	60 days delinquent
            5: (0, 0, 0, 0, 0, 1, 0, 0, 0),
            # 6	30 days delinquent
            6: (0, 0, 0, 0, 1, 0, 0, 0, 0),
            # 7	unknown delinquency
            7: (0, 0, 0, 1, 0, 0, 0, 0, 0),
            # 8	current and never delinquent
            8: (0, 0, 1, 0, 0, 0, 0, 0, 0),
            # 9	all other
            9: (0, 1, 0, 0, 0, 0, 0, 0, 0),
            # XXX NOT DOCUMENTED
            -9: (1, 0, 0, 0, 0, 0, 0, 0, 0),
        }
        X2 = np.array([to_onehot[x] for x in list(df['MaxDelqEver'])])

        df.drop(columns=['RiskPerformance', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver'], inplace=True)
        X = np.hstack([df.to_numpy(), X1, X2])
        X = pd.DataFrame(data=X, columns=list(map(str, range(X.shape[1]))))

        sampled_idx, _ = list(StratifiedShuffleSplit(n_splits=2,
                                                     train_size=0.2,
                                                     random_state=0).split(X, y))[0]
        X, y = X.iloc[sampled_idx], y[sampled_idx]

        super().__init__(model, X, y,
                         feature_names=X.columns,
                         name="Heloc",
                         metric="f1",
                         prop_known=0.001,
                         rng=rng,
                         normalizer=Normalizer())
