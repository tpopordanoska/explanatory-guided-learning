import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer

from .experiment import Experiment
from src.models.learners import *


class Newsgroups(Experiment):
    def __init__(self, **kwargs):
        rng = kwargs.pop("rng")
        kind = kwargs.pop("kind")
        model = GradientBoosting()

        dataset = fetch_20newsgroups(data_home='data',
                                     subset='all',
                                     remove=('headers', 'footers', 'quotes'))

        relabel = [
            [1, 0, 0, 0, 0], #'alt.atheism': 0,
            [0, 1, 0, 0, 0], #'comp.graphics': 0,
            [0, 1, 0, 0, 0], #'comp.os.ms-windows.misc': 0,
            [0, 1, 0, 0, 0], #'comp.sys.ibm.pc.hardware': 0,
            [0, 1, 0, 0, 0], #'comp.sys.mac.hardware': 0,
            [0, 1, 0, 0, 0], #'comp.windows.x': 0,
            [0, 0, 0, 0, 0], #'misc.forsale': 1,
            [0, 0, 1, 0, 0], #'rec.autos': 1,
            [0, 0, 1, 0, 0], #'rec.motorcycles': 1,
            [0, 0, 1, 0, 0], #'rec.sport.baseball': 0,
            [0, 0, 1, 0, 0], #'rec.sport.hockey': 0,
            [0, 0, 0, 1, 0], #'sci.crypt': 0,
            [0, 0, 0, 1, 0], #'sci.electronics': 1,
            [0, 0, 0, 1, 0], #'sci.med': 1,
            [0, 0, 0, 1, 0], #'sci.space': 0,
            [1, 0, 0, 0, 0], #'soc.religion.christian': 0,
            [0, 0, 0, 0, 1], #'talk.politics.guns': 0,
            [0, 0, 0, 0, 1], #'talk.politics.mideast': 0,
            [0, 0, 0, 0, 1], #'talk.politics.misc': 0,
            [1, 0, 0, 0, 0], #'talk.religion.misc': 0,
        ]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataset.data).astype(np.int8).todense() # (18846, 134410)
        y = np.vectorize(lambda label: relabel[label][kind])(dataset.target)
        sampled_idx, _ = list(StratifiedShuffleSplit(n_splits=2,
                                                     train_size=0.1,
                                                     random_state=0).split(X, y))[0]
        X, y = X[sampled_idx], y[sampled_idx] # (3769, 134410)
        X = RFE(LinearSVC(random_state=0),
                step=10000).fit_transform(X, y) # (3769, 67205)
        jndices = list(range(X.shape[1]))
        X = X[:, np.random.RandomState(0).choice(jndices, size=10000)]
        X = pd.DataFrame(data=X, columns=[f"c{j}" for j in range(X.shape[1])])

        super().__init__(model, X, y,
                         feature_names=X.columns,
                         name=f"20 Newsgroups {kind}",
                         prop_known=0.001,
                         normalizer=Normalizer(),
                         rng=rng)


class NewsgroupsReligion(Newsgroups):
    def __init__(self, **kwargs):
        kwargs["kind"] = 0
        super().__init__(**kwargs)


class NewsgroupsScience(Newsgroups):
    def __init__(self, **kwargs):
        kwargs["kind"] = 3
        super().__init__(**kwargs)
