from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment


class BreastCancer(Experiment):

    def __init__(self, **kwargs):
        model = kwargs.pop("model")
        rng = kwargs.pop("rng")

        # Samples per class	212(M),357(B)
        dataset = load_breast_cancer()

        X = dataset.data
        # 0 is "malignant", 1 is "benign"
        y = dataset.target

        super().__init__(model, X, y, feature_names=list(dataset.feature_names), name="Breast Cancer", prop_known=0.01,
                         rng=rng, normalizer=StandardScaler())
