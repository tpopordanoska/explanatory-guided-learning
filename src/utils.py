import os
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from .experiments import *
from .normalizer import *


def create_folders():
    """
    Create folders for storing the plots and the graphs.

    :return: The path to the created folder
    """
    path_results = "{}\\results".format(os.getcwd())
    try:
        os.mkdir(path_results)
    except FileExistsError:
        print("Directory {} already exists".format(path_results))
    except OSError:
        print("Creation of the directory {} failed".format(path_results))
    else:
        print("Successfully created the directory {} ".format(path_results))

    # Create a separate folder for each time running the experiment
    path = "{}\\{}". format(path_results, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f'))
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory ", path, " already exists")
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    return path


def create_experiment_folder(path, experiment):
    """
    Create folder for storing results for the given experiment and model

    :param path: Path to the folder created when the script is run
    :param experiment: The name of the experiment being performed
    :return: model: The name of the model currently running
    """
    path_experiment = "{}\\{}".format(path, experiment)
    try:
        os.mkdir(path_experiment)
    except OSError:
        print("Creation of the directory %s failed" % path_experiment)
    return path_experiment


def create_model_folder(path, model):

    path_model = "{}\\{} {}".format(path, model, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.mkdir(path_model)
    except OSError:
        print("Creation of the directory %s failed" % path_model)

    return path_model


def get_mean_and_std(scores_dict, n_folds):
    if not scores_dict:
        return {}, {}
    scores_dict_mean = {}
    scores_dict_std = {}
    for method, scores in scores_dict.items():
        if scores:
            smallest_len = min([len(x) for x in scores])
            scores_smallest_len = [s[:smallest_len] for s in scores]
            scores_dict_mean[method] = np.mean(scores_smallest_len, axis=0)
            scores_dict_std[method] = np.std(scores_smallest_len, axis=0) / np.sqrt(n_folds)

    return scores_dict_mean, scores_dict_std


def get_passive_score(experiment, file, n_splits, split_seed, scorer):
    pipeline = Pipeline([('transformer', experiment.normalizer), ('estimator', experiment.model._model)])
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    scores = cross_val_score(pipeline, experiment.X, experiment.y, cv=kfold, scoring=scorer)
    scores_mean_std = {
        "mean": np.mean(scores, axis=0),
        "std": np.std(scores, axis=0) / np.sqrt(n_splits)
    }
    file.write("(sklearn) Passive {} for {}: {:.2f} (+/- {:.2f}) ".format(scorer, experiment.name, scores.mean(),
                                                                          scores.std() * 2))
    file.write("Passive {} for {}: {:.2f} (+/- {:.2f}) ".format(scorer, experiment.name, scores_mean_std["mean"],
                                                                scores_mean_std["std"]))
    return scores_mean_std


def random_sampling(**kwargs):
    experiment = kwargs.pop("experiment")
    train_idx = kwargs.pop("train_idx")
    return select_random(train_idx, experiment.model.rng)


def least_confident_idx(**kwargs):
    """
    Get the index of the example closest to the decision boundary.

    :param kwargs: Keyword arguments

    :return: The index (in X) of the least confident example
    """
    experiment = kwargs.pop("experiment")
    known_idx = kwargs.pop("known_idx")
    train_idx = kwargs.pop("train_idx")
    X_known = get_from_indexes(experiment.X, known_idx)
    X_train = get_from_indexes(experiment.X, train_idx)

    _, X_train_norm = Normalizer(experiment.normalizer).normalize_known_train(X_known, X_train)

    if hasattr(experiment.model._model, "decision_function"):
        margins = np.abs(experiment.model.decision_function(X_train_norm))
    elif hasattr(experiment.model._model, "predict_proba"):
        probs = experiment.model.predict_proba(X_train_norm)
        margins = np.sum(probs * np.log(probs), axis=1).ravel()
    else:
        raise AttributeError("Model with either decision_function or predict_proba method")

    if len(margins) == 0:
        return None
    return train_idx[np.argmin(margins)]


def get_from_indexes(X, indexes):
    if isinstance(X, pd.DataFrame):
        return X.iloc[indexes]
    return X[indexes]


def select_by_coordinates(x, y, data):
    """
    Get the index of the element in the data, if found

    :param x: The x coordinate
    :param y: The y coordinate
    :param data: The data to find the element by coordinates from
    :return: The index of the found element
    """
    # TODO: take care of element not found and fix [][]
    return [np.where((data[:, 0] == x) & (data[:, 1] == y))[0][0]]


def select_random(data, rng):
    """
    Get a random element from the given data.

    :param data: The data to find a random element from
    :param rng: RandomState object

    :return: A random element from the data
    """
    return rng.choice(data)


def concatenate_data(X_train, y_train, X_unlabeled, y_unlabeled, y_pred):
    """
    Concatenate the given data in one matrix with three columns.

    :param X_train: The coordinates of the points of the first data matrix
    :param y_train: The labels of the points from the first data matrix
    :param X_unlabeled: The coordinates of the points of the second data matrix
    :param y_unlabeled: The labels of the points from the second data matrix
    :param y_pred: The predictions of the unlabeled points

    :return: One matrix containing the features, the predictions and the true labels
    """
    Xy_train = np.concatenate((X_train, np.array([y_train]).T), axis=1)
    Xy_unlabeled = np.concatenate((X_unlabeled, np.array([y_pred]).T), axis=1)
    Xy = np.concatenate((Xy_train, Xy_unlabeled), axis=0)
    true_labels = np.concatenate((y_train, y_unlabeled))
    return np.concatenate((Xy, np.array([true_labels]).T), axis=1)