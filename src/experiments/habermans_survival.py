import pandas as pd
from sklearn.utils import check_random_state

from .experiment import Experiment


class HabermansSurvival(Experiment):

    def __init__(self, rng):

        rng = check_random_state(rng)

        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"]
        self.load_dataset('data', urls)

        columns = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']
        dataset = pd.read_csv("data/haberman.data", names=columns)

        # target values: 1 = the patient survived 5 years or longer, 2 (changed to 0) = the patient died within 5 years
        y = dataset['Survival status'].map({1: 1, 2: 0}).to_numpy()

        # creating the feature vector
        X = dataset.drop('Survival status', axis=1)

        # sc = StandardScaler()
        # X_processed = sc.fit_transform(X)

        super().__init__(X, y, feature_names=list(X.columns.values), name="Banknote Auth", prop_known=0.1, rng=rng)
