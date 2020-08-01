import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils.arguments import get_experiment_args
from src.utils.common import create_folders, initialize_experiment
from src.utils.normalizer import Normalizer
from src.utils.plotting import save_plot


def main(args):
    path_results = create_folders()
    for experiment_name in args.experiments:
        experiment = initialize_experiment(experiment_name, args.seed, path_results,)

        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=experiment.rng)
        for k, (train_idx, test_idx) in enumerate(kfold.split(experiment.X, experiment.y)):
            X_train_norm, X_test_norm = get_normalized_features(experiment, train_idx, test_idx)
            train(experiment.model, X_train_norm, experiment.y[train_idx])
            y_pred, probs_test = get_predictions_and_probs(experiment.model, X_test_norm)
            entropy = calculate_entropy_of_mistakes(experiment.y[test_idx], y_pred, probs_test)
            plot_and_save_histogram(entropy, experiment.path)


def get_normalized_features(experiment, train_idx, test_idx):
    X_train, X_test = get_from_indexes(experiment.X, train_idx), get_from_indexes(experiment.X, test_idx)
    return Normalizer(experiment.normalizer).normalize_known_train(X_train, X_test)


def train(model, features, labels):
    model.fit(features, labels)


def get_predictions_and_probs(model, X_test_norm):
    y_pred = model.predict(X_test_norm)
    probs = model.predict_proba(X_test_norm)

    return y_pred, probs


def calculate_entropy_of_mistakes(y_test, y_pred, probs_test):
    mistakes_idx = np.where(y_pred != y_test)[0]
    mistakes_probs = probs_test[mistakes_idx]
    entropy = - np.sum(mistakes_probs * np.log(mistakes_probs), axis=1).ravel()

    return entropy


def plot_and_save_histogram(entropy, path):
    plt.hist(entropy, bins=10)
    plt.title("# Mistakes: {}".format(len(entropy)))
    plt.xlabel("Entropy")
    plt.ylabel("# Mistakes")

    save_plot(plt, path, "Entropy")


def get_from_indexes(X, indexes):
    if isinstance(X, pd.DataFrame):
        return X.iloc[indexes]
    return X[indexes]

if __name__ == '__main__':
    exp_args = get_experiment_args()
    main(exp_args)
