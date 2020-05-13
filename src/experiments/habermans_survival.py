import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .experiment import Experiment
from ..learners import *


class HabermansSurvival(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = SVM(name='SVM (gamma=10, C=10)', gamma=10, C=10)

        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"]
        self.load_dataset('data', urls)

        columns = ['Age', 'Year_operation', 'Axillary_nodes_detected', 'Survival_status']
        dataset = pd.read_csv("data/haberman.data", names=columns)

        # target values: 255 1s (chnaged to 0) = the patient survived 5 years or longer, and 81 2s (changed to 1) = the patient died within 5 years
        y = dataset['Survival_status'].map({1: 0, 2: 1}).to_numpy()

        # creating the feature vector
        X = dataset.drop('Survival_status', axis=1)

        super().__init__(model, X.to_numpy(), y, feature_names=list(X.columns.values), name="Haberman's Survival",
                         prop_known=0.01, rng=rng, normalizer=MinMaxScaler())
