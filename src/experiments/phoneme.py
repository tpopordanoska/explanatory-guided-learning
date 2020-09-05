import pandas as pd
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment
from src.models.learners import *


class Phoneme(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        columns = ['Aa', 'Ao', 'Dcl', "Iy", 'Sh', 'Class']
        dataset = pd.read_csv("data/phoneme.dat", skiprows=10, names=columns, delimiter=',')

        # Class0: 3,818 samples, Class1: 1,586
        y = dataset['Class'].to_numpy()
        X = dataset.drop('Class', axis=1)

        super().__init__(model, X, y, feature_names=X.columns, name="Phoneme",
                         prop_known=0.001, rng=rng, normalizer=StandardScaler())
