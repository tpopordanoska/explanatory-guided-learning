import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment
from src.models.learners import *


class German(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"]
        self.load_dataset('data', urls)

        columns = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']
        global dataset
        dataset = pd.read_csv("data/german.data", delimiter=' ', skiprows=1, names=columns)

        num_pipeline = Pipeline(steps=[
            ("num_attr_selector", ColumnsSelector(type='int64')),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline(steps=[
            ("cat_attr_selector", ColumnsSelector(type='object')),
            ("encoder", CategoricalEncoder(dropFirst=True))
        ])
        full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])

        # Binarize the y output for easier use -> 0 = 'bad' credit; 1 (700) = 'good' credit
        # target values: 700 1s (changed to 0) = Good, 300 2s (changed to 1) = Bad
        y = dataset['classification'].map({1: 0, 2: 1}).to_numpy()

        # creating the feature vector
        X = dataset.drop('classification', axis=1)

        super().__init__(model, X, y, feature_names=X.columns, name="German", metric="f1",
                         prop_known=0.01, rng=rng, normalizer=full_pipeline)


class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(data=X)
        return X.select_dtypes(include=[self.type])


class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, strategy='most_frequent'):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns

        if self.strategy is 'most_frequent':
            self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
        else:
            self.fill = {column: '0' for column in self.columns}

        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].fillna(self.fill[column])
        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, dropFirst=True):
        self.categories = dict()
        self.dropFirst = dropFirst

    def fit(self, X, y=None):
        join_df = dataset
        join_df = join_df.select_dtypes(include=['object'])
        for column in join_df.columns:
            self.categories[column] = join_df[column].value_counts().index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.select_dtypes(include=['object'])
        for column in X_copy.columns:
            X_copy[column] = X_copy[column].astype({column: CategoricalDtype(self.categories[column])})
        return pd.get_dummies(X_copy, drop_first=self.dropFirst)