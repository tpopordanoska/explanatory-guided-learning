import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.learners import *
from .experiment import Experiment


class HabermansSurvival(Experiment):

    def __init__(self, rng):

        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"]
        self.load_dataset('data', urls)

        columns = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']
        dataset = pd.read_csv("data/haberman.data", names=columns)

        # target values: 255 1s = the patient survived 5 years or longer, and 81 2s (changed to 0) = the patient died within 5 years
        y = dataset['Survival status'].map({1: 1, 2: 0}).to_numpy()

        # creating the feature vector
        X = dataset.drop('Survival status', axis=1)

        sc = MinMaxScaler()
        X_processed = sc.fit_transform(X)

        model = SVM(name='svm_rbf', rng=rng, gamma=1, C=1e2)

        super().__init__(model, X_processed, y, feature_names=list(X.columns.values), name="Banknote Auth", prop_known=0.01, rng=rng)
