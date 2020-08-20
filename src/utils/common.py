import os
import pickle
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from src.utils.constants import EXPERIMENTS

from src.experiments import *


def create_folders():
    """
    Create folders for storing the plots and the graphs.

    :return: The path to the created folder
    """
    path_results = os.path.join(os.getcwd(), "results")
    try:
        os.mkdir(path_results)
    except FileExistsError:
        print("Directory {} already exists".format(path_results))
    except OSError:
        print("Creation of the directory {} failed".format(path_results))
    else:
        print("Successfully created the directory {} ".format(path_results))

    # Create a separate folder for each time running the experiment
    path = os.path.join(path_results, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f'))
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory ", path, " already exists")
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    return path


def create_folder(path, folder_name):
    """
    Create folder for storing results for the given experiment and model

    :param path: Path to the folder to be created 
    :param folder_name: The name of the folder to be created

    :return: path_folder: The path to the created folder
    """
    path_folder = os.path.join(path, folder_name)
    if not os.path.exists(path_folder):
        try:
            os.mkdir(path_folder)
        except OSError:
            print("Creation of the directory %s failed" % path_folder)
    return path_folder
 
 
def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, data, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, **kwargs)


def initialize_experiment(experiment_name, seed, path_results):
    print(experiment_name)
    experiment = EXPERIMENTS[experiment_name](rng=np.random.RandomState(seed))
    experiment.path = create_folder(path_results, experiment_name)
    experiment.file = open(os.path.join(experiment.path, 'out.txt'), 'w')

    return experiment


def create_strategies_list(args):
    methods = []
    if "al_dw" in args.strategies:
        args.strategies.remove("al_dw")
        for beta in args.betas:
            methods.append("al_dw_{}".format(beta))
    if "xgl_rules" in args.strategies:
        args.strategies.remove("xgl_rules")
        for theta_rules in args.thetas_rules:
            methods.append("xgl_rules_{}".format(theta_rules))
    if "xgl_rules_hierarchy" in args.strategies:
        args.strategies.remove("xgl_rules_hierarchy")
        for theta_rules in args.thetas_rules:
            methods.append("xgl_rules_hierarchy_{}".format(theta_rules))
    if "xgl_rules_simple_tree" in args.strategies:
        args.strategies.remove("xgl_rules_simple_tree")
        for theta_rules in args.thetas_rules:
            methods.append("xgl_rules_simple_tree_{}".format(theta_rules))
    if "xgl_clusters" in args.strategies:
        args.strategies.remove("xgl_clusters")
        for theta_xgl in args.thetas_xgl:
            methods.append("xgl_clusters_{}".format(theta_xgl))
    return methods + args.strategies


def write_to_file(loop, args, k, experiment, method, exec_time):
    file = experiment.file
    known_idx = loop.initial_known_idx
    train_idx = loop.initial_train_idx
    test_idx = loop.initial_test_idx

    # Write parameters to file
    file.write('seed, fold {} : #known {}, #train {}, #test {} \n'
               .format(args.seed, k + 1, len(known_idx), len(train_idx), len(test_idx)))
    _, counts_known = np.unique(experiment.y[known_idx], return_counts=True)
    _, counts_train = np.unique(experiment.y[train_idx], return_counts=True)
    _, counts_test = np.unique(experiment.y[test_idx], return_counts=True)
    file.write("Known bincount: {}\n".format(counts_known))
    file.write("Train bincount: {}\n".format(counts_train))
    file.write("Test bincount: {}\n".format(counts_test))

    print(method)
    file.write("Method: {} \n".format(method))
    file.write("Model: {}\n".format(experiment.model.sklearn_model))
    file.write("{} clusters, {} folds, {} seed\n".format(args.n_clusters, args.n_folds, args.seed))
    file.write("Execution time: {} \n".format(exec_time))


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


def get_passive_score(running_instance, scorer):
    exp = running_instance.experiment
    pipeline = Pipeline([('transformer', exp.normalizer), ('estimator', exp.model.sklearn_model)])
    kfold = StratifiedKFold(n_splits=running_instance.args.n_folds, shuffle=True, random_state=exp.rng)
    scores = cross_val_score(pipeline, exp.X, exp.y, cv=kfold, scoring=scorer)
    scores_mean_std = {
        "mean": np.mean(scores, axis=0),
        "std": np.std(scores, axis=0) / np.sqrt(running_instance.args.n_folds)
    }
    exp.file.write("(sklearn) Passive {} for {}: {:.2f} (+/- {:.2f}) ".format(
        scorer, exp.name, scores.mean(), scores.std() * 2))
    exp.file.write("Passive {} for {}: {:.2f} (+/- {:.2f}) ".format(
        scorer, exp.name, scores_mean_std["mean"], scores_mean_std["std"]))

    return scores_mean_std


def extract_param_from_name(method):
    # Get the theta value for XGL and rules or beta value for density based AL
    param = ""
    if "xgl" in method or "rules" in method or "dw" in method:
        param = float(method.split("_")[-1])
        method = "_".join(method.split("_")[:-1])
    return param, method

