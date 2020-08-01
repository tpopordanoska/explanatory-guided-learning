import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment
from src.models.learners import *


class Credit(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls"]
        self.load_dataset('data', urls)

        data = pd.read_excel("data/default of credit card clients.xls", header=1)
        data.drop('ID', axis=1, inplace=True)
        dataset = Bunch(
            target=np.array(data['default payment next month']),
            data=(data.drop('default payment next month', axis=1))
        )

        # total: 30000 instances, bincount y: [23364,  6636]
        X = dataset.data
        # 0 = 'not default', 1 = 'default'
        y = dataset.target

        # Categorical features: SEX(1=male, 2=female), EDUCATION(1 = graduate school, 2 = university, 3 = high school
        #  4 = others, 5 = unknown, 6 = unknown, 0 = unknown), MARRIAGE(1 = married, 2 = single, 3 = others),
        #  PAY_0,2,3,4,5,6 (-2= no consumption, -1 = pay duly, 1 = payment delay for one month,
        #  2 = payment delay for two months ... 9 = payment delay for nine months and above

        # Most important features (RFE)
        Ximp = X[['PAY_0', 'BILL_AMT1', 'PAY_AMT2']].astype('float')

        sampled_idx, _ = list(StratifiedShuffleSplit(n_splits=2, train_size=0.1, random_state=0).split(X, y))[0]
        X, y = Ximp.iloc[sampled_idx], y[sampled_idx]

        super().__init__(model, X, y, feature_names=list(X.columns.values), name="Credit Card Default",
                         prop_known=0.001, rng=rng, normalizer=StandardScaler())
