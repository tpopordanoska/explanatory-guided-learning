import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.learners import *
from .experiment import Experiment


class BanknoteAuth(Experiment):

    def __init__(self, rng):

        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"]
        self.load_dataset('data', urls)

        # total 1372 instances, 610 1s = fake, 762 0s = real
        columns = ["variance", "skew", "curtosis", "entropy", "class"]
        dataset = pd.read_csv("data/data_banknote_authentication.txt", names=columns)

        # target values: 1 is fake, 0 is real
        y = dataset['class'].to_numpy()

        # creating the feature vector
        X = dataset.drop('class', axis=1)

        sc = MinMaxScaler()
        X_processed = sc.fit_transform(X)

        model = SVM(name='svm_rbf', rng=rng, gamma=1, C=1e2)

        super().__init__(model, X_processed, y, feature_names=list(X.columns.values), name="Banknote Auth", prop_known=0.001, rng=rng)
