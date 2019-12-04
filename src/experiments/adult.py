import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from .experiment import Experiment


class Adult(Experiment):

    def __init__(self, rng):

        rng = check_random_state(rng)

        urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]
        self.load_dataset('data', urls)

        columns = ["age", "workClass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        train_data = pd.read_csv('data/adult.data', names=columns, sep=' *, *', na_values='?')
        test_data = pd.read_csv('data/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')
        dataset = pd.concat([train_data, test_data])

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
        # pass the data through the full_pipeline
        X_processed = full_pipeline.fit_transform(X)

        column_names = ['Col_' + str(i) for i in range(0, X_processed.shape[1])]

        super().__init__(X_processed, y, feature_names=column_names, name="Adult", prop_known=0.01, rng=rng)


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
        join_df = X # pd.concat([train_data, test_data])
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
