from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from .experiment import Experiment


class BreastCancer(Experiment):

    def __init__(self, rng):

        rng = check_random_state(rng)

        # Samples per class	212(M),357(B)
        # Samples total	569
        # Dimensionality	30
        dataset = load_breast_cancer()

        X = dataset.data
        # 0 is "malignant", 1 is "benign"
        y = dataset.target

        sc = StandardScaler()
        X = sc.fit_transform(X)

        super().__init__(X, y, feature_names=list(dataset.feature_names), name="Breast Cancer", prop_known=0.05, rng=rng)
