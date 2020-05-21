import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment
from ..learners import *


class Adult(Experiment):

    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        model = GradientBoosting()

        urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]
        self.load_dataset('data', urls)

        columns = ["age", "workClass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        global train_data
        train_data = pd.read_csv('data/adult.data', names=columns, sep=' *, *', na_values='?')
        test_data = pd.read_csv('data/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')
        dataset = train_data # pd.concat([train_data, test_data])

        num_pipeline = Pipeline(steps=[
            ("num_attr_selector", ColumnsSelector(type='int64')),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline(steps=[
            ("cat_attr_selector", ColumnsSelector(type='object')),
            ("cat_imputer", CategoricalImputer(columns=['workClass', 'occupation', 'native-country'])),
            ("encoder", CategoricalEncoder(dropFirst=True))
        ])
        full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])

        dataset.drop(['fnlwgt', 'education'], axis=1, inplace=True)
        # convert the income column to 0 or 1 and then drop the column for the feature vectors
        dataset["income"] = dataset["income"].apply(lambda x: 0 if x == '<=50K' else 1)
        # target values
        y = dataset['income'].to_numpy()
        # creating the feature vector
        X = dataset.drop('income', axis=1)

        sampled_idx, _ = list(StratifiedShuffleSplit(n_splits=2, train_size=0.1, random_state=0).split(X, y))[0]
        X, y = X.iloc[sampled_idx], y[sampled_idx]

        column_names = ['Col_' + str(i) for i in range(0, X.shape[1])]

        #After resampling with train_size = 0.1: total 3256 examples, 784 1s, 2472 0s
        super().__init__(model, X, y, feature_names=column_names, name="Adult", prop_known=0.001, rng=rng,
                         normalizer=full_pipeline)


class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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
        join_df = train_data # pd.concat([train_data, test_data])
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
