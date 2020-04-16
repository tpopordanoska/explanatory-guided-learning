import os
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from .experiments import *


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


def get_from_indexes(X, indexes):
    if isinstance(X, pd.DataFrame):
        return X.iloc[indexes]
    return X[indexes]


def select_random(data, rng):
    """
    Get a random element from the given data.

    :param data: The data to find a random element from
    :param rng: RandomState object

    :return: A random element from the data
    """
    return rng.choice(data)
