from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

from src.learners import *
from .experiment import Experiment


class BreastCancer(Experiment):

    def __init__(self, rng):

        # Samples per class	212(M),357(B)
        dataset = load_breast_cancer()

        X = dataset.data
        # 0 is "malignant", 1 is "benign"
        y = dataset.target

        sc = MinMaxScaler()
        X_processed = sc.fit_transform(X)

        model = SVM(name='svm_rbf', rng=rng, gamma=1, C=1e2)

        super().__init__(model, X_processed, y, feature_names=list(dataset.feature_names), name="Breast Cancer", prop_known=0.01, rng=rng)
