from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment
from src.models.learners import *


class BreastCancer(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = SVM(name="SVM (gamma=0.01, C=100)", gamma=0.01, C=100)

        # Samples per class	212(M),357(B)
        dataset = load_breast_cancer()

        X = dataset.data
        # 0 is "malignant", 1 is "benign"
        y = dataset.target
        feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
                         'mean_smoothness', 'mean_compactness', 'mean_concavity',
                         'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
                         'radius_error', 'texture_error', 'perimeter_error', 'area_error',
                         'smoothness_error', 'compactness_error', 'concavity_error',
                         'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
                         'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
                         'worst_smoothness', 'worst_compactness', 'worst_concavity',
                         'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']

        super().__init__(model, X, y, feature_names=feature_names, name="Breast Cancer", prop_known=0.01,
                         rng=rng, normalizer=StandardScaler())
