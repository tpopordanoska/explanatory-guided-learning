import pandas as pd
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment
from src.models.learners import *


class SteelPlateFaults(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"]
        self.load_dataset('data', urls)

        columns = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
                   "Sum_of_Luminosity",	"Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer",
                   "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
                   "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index",
                   "LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index", "Luminosity_Index",
                   "SigmoidOfAreas", "Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
        dataset = pd.read_csv("data/faults.nna", names=columns, delimiter='\t')

        # Drop OHE of target types and create a single Target column
        targets = dataset.iloc[:, 27:35]
        dataset.drop(targets.columns, axis=1, inplace=True)
        dataset['Target'] = targets.idxmax(1)

        # value counts: Dirtiness 55, Stains 72, Pastry 158, Z_Scratch 190, K_scratch 391, Bumps 402, Other_Faults 673
        y = dataset['Target'].map({
            'Dirtiness': 1,  # rare class
            'Stains': 1,
            'Pastry': 1,
            'Z_Scratch': 0,
            'K_Scratch': 0,
            'Bumps': 0,
            'Other_Faults': 0}).to_numpy()

        # Total data points: 1941
        # TypeOfSteel_A300, TypeOfSteel_A400 and Outside_Global_Index are categorical
        X = dataset.drop('Target', axis=1)

        super().__init__(model, X, y, feature_names=X.columns, name="Steel Plate Faults",
                         prop_known=0.001, rng=rng, normalizer=StandardScaler())
