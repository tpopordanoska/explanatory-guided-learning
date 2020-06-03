import argparse
import os

import numpy as np

from src.utils import load, get_mean_and_std

RESULT_TABLE = ["al_density_weighted_1", "al_least_confident", "sq_random", "rules_100.0"]
FORMATTING = {
    'adult': 'adult     ',
    'australian': 'australian',
    'banknote-auth': 'banknote  ',
    'breast-cancer': 'cancer    ',
    'credit': 'credit    ',
    'german': 'german    ',
    'heart': 'heart     ',
    'hepatitis': 'hepatitis ',
    'synthetic': 'synthetic '
}


def create_results_table(pickle_files, result_file, path, folder):
    final_string = ""
    for filename in pickle_files:
        results = load(os.path.join(path, folder, filename))
        test_f1_dict = results["test_f1"]
        queries_f1_dict = results["queries_f1"]
        n_folds = results["args"]["n_folds"]

        test_dict_mean, _ = get_mean_and_std(test_f1_dict, n_folds)
        queries_dict_mean, _ = get_mean_and_std(queries_f1_dict, n_folds)

        experiment_string = FORMATTING[filename.split("__")[0]]
        for strategy in RESULT_TABLE:
            avg_f1 = np.mean(test_dict_mean[strategy])
            std_f1 = np.std(test_dict_mean[strategy])
            narrative_bias = narrative_bias_mean(test_dict_mean[strategy], queries_dict_mean[strategy])
            experiment_string += " & ${:.2f} \pm {:.2f}$ & {:.2f}".format(avg_f1, std_f1, narrative_bias)
        final_string += experiment_string + " \\\\ \n"

    result_file.write(final_string)


def narrative_bias_mean(test_scores, query_scores):
    score_ma = running_mean(query_scores, 20)
    return np.mean(score_ma - test_scores[:len(score_ma)])


def running_mean(data, window_width):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_width:] - cumsum[:-window_width]) / float(window_width)


if __name__ == '__main__':
    path_results = os.path.join(os.getcwd(), "results")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder',
                        type=str,
                        default=os.listdir(path_results)[-1],
                        help="The name of the folder where the .pickle files are stored")

    args = parser.parse_args()

    results_folder = os.path.join(path_results, args.folder)
    file = open(os.path.join(results_folder, 'final_results.txt'), 'w')
    pickles = [f for f in os.listdir(results_folder) if f.endswith(".pickle")]

    create_results_table(pickles, file, path_results, args.folder)
