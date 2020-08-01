import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .experiment import Experiment
from src.models.learners import *


class Heart(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        url = ["http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"]
        self.load_dataset('data', url)

        columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                   'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression',
                   'st_slope', 'num_major_vessels', 'thalassemia', 'target']

        dataset = pd.read_csv("data/processed.cleveland.data", delimiter=',', names=columns)

        # 0 = absence, 1, 2, 3, 4 = present
        y = dataset['target'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1}).to_numpy()
        X = dataset.drop('target', axis=1)

        X = X.replace({'?': np.nan})
        X.fillna(X.median(), inplace=True)

        # source: https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model
        X["num_major_vessels"] = X["num_major_vessels"].map({'0.0': '0', '1.0': '1', '2.0': '2', '3.0': '3'})
        X["thalassemia"] = X["thalassemia"].map({'3.0': 3, '6.0': 6, '7.0': 7})

        X['sex'][X['sex'] == 0] = 'female'
        X['sex'][X['sex'] == 1] = 'male'

        X['chest_pain_type'][X['chest_pain_type'] == 1] = 'typical_angina'
        X['chest_pain_type'][X['chest_pain_type'] == 2] = 'atypical_angina'
        X['chest_pain_type'][X['chest_pain_type'] == 3] = 'non_anginal_pain'
        X['chest_pain_type'][X['chest_pain_type'] == 4] = 'asymptomatic'

        X['fasting_blood_sugar'][X['fasting_blood_sugar'] == 0] = 'lower_120mgml'
        X['fasting_blood_sugar'][X['fasting_blood_sugar'] == 1] = 'greater_120mgml'

        X['rest_ecg'][X['rest_ecg'] == 0] = 'normal'
        X['rest_ecg'][X['rest_ecg'] == 1] = 'STT_wave_abnormality'
        X['rest_ecg'][X['rest_ecg'] == 2] = 'left_ventricular_hypertrophy'

        X['exercise_induced_angina'][X['exercise_induced_angina'] == 0] = 'no'
        X['exercise_induced_angina'][X['exercise_induced_angina'] == 1] = 'yes'

        X['st_slope'][X['st_slope'] == 1] = 'upsloping'
        X['st_slope'][X['st_slope'] == 2] = 'flat'
        X['st_slope'][X['st_slope'] == 3] = 'downsloping'

        X['thalassemia'][X['thalassemia'] == 3] = 'normal'
        X['thalassemia'][X['thalassemia'] == 6] = 'fixed_defect'
        X['thalassemia'][X['thalassemia'] == 7] = 'reversable_defect'

        X['sex'] = X['sex'].astype('object')
        X['chest_pain_type'] = X['chest_pain_type'].astype('object')
        X['fasting_blood_sugar'] = X['fasting_blood_sugar'].astype('object')
        X['rest_ecg'] = X['rest_ecg'].astype('object')
        X['exercise_induced_angina'] = X['exercise_induced_angina'].astype('object')
        X['st_slope'] = X['st_slope'].astype('object')
        X['thalassemia'] = X['thalassemia'].astype('object')

        X = pd.get_dummies(X, drop_first=True)

        super().__init__(model, X, y, feature_names=X.columns, name="Heart", metric="f1",
                         prop_known=0.001, rng=rng, normalizer=MinMaxScaler())
