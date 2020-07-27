import os
import pickle
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from constants import EXPERIMENTS

from .experiments import *


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


def save_plot(plt, path, img_name, plot_title=None, use_grid=True, use_date=True):
    img_name = "{}.pdf".format(img_name)
    plt.grid(use_grid)
    if plot_title is not None:
        plt.title(plot_title)
    if path:
        try:
            if use_date:
                img_name = "{}-{}.pdf".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f'), img_name)
            plt.savefig(os.path.join(path, img_name), bbox_inches='tight', format='pdf')
        except ValueError:
            print("Something went wrong while saving image: ", img_name)
    else:
        plt.show()
    plt.close()


def initialize(experiment_name, path_results, seed):
    print(experiment_name)
    experiment = EXPERIMENTS[experiment_name](rng=np.random.RandomState(seed))
    path = create_folder(path_results, experiment_name)
    file = open(os.path.join(path, 'out.txt'), 'w')
    return experiment, path, file


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
    if "xgl_clusters" in args.strategies:
        args.strategies.remove("xgl_clusters")
        for theta_xgl in args.thetas_xgl:
            methods.append("xgl_clusters_{}".format(theta_xgl))
    return methods + args.strategies


def write_to_file(file, known_idx, train_idx, test_idx, seed, k, experiment,
                  method, n_clusters, n_folds, use_weights, use_labels):
    # Write parameters to file
    file.write('seed, fold {} : #known {}, #train {}, #test {} \n'
               .format(seed, k + 1, len(known_idx), len(train_idx), len(test_idx)))
    _, counts_known = np.unique(experiment.y[known_idx], return_counts=True)
    _, counts_train = np.unique(experiment.y[train_idx], return_counts=True)
    _, counts_test = np.unique(experiment.y[test_idx], return_counts=True)
    file.write("Known bincount: {}\n".format(counts_known))
    file.write("Train bincount: {}\n".format(counts_train))
    file.write("Test bincount: {}\n".format(counts_test))

    print(method)
    file.write("Method: {} \n".format(method))
    file.write("Model: {}\n".format(experiment.model.sklearn_model))
    file.write("{} clusters, {} folds, {} seed\n".format(n_clusters, n_folds, seed))
    file.write("use_weights={}, use_labels={}\n".format(use_weights, use_labels))


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
    pipeline = Pipeline([('transformer', experiment.normalizer), ('estimator', experiment.model.sklearn_model)])
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
