import pandas as pd
from sklearn.preprocessing import StandardScaler

from .experiment import Experiment


class Australian(Experiment):

    def __init__(self, **kwargs):
        model = kwargs.pop("model")

        urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"]
        self.load_dataset('data', urls)

        columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'Y']
        dataset = pd.read_table("data/australian.dat", sep='\s+', header=None, names=columns)

        # 6 numerical and 8 categorical attributes
        # df_cat = dataset[['X1', 'X4', 'X5', 'X6', 'X8', 'X9', 'X11', 'X12', 'Y']]
        # df_cont = dataset[['X2', 'X3', 'X7', 'X10', 'X13', 'X14', 'Y']]

        y = dataset['Y'].to_numpy()
        X = dataset.drop('Y', axis=1)

        super().__init__(model, X, y, feature_names=X.columns, name="Australian", metric="f1",
                         prop_known=0.001, rng=model.rng, normalizer=StandardScaler())

