import argparse

from constants import *
from src import *


def plot_results(scores_dict, scores_test_dict, score_passive, path, scorer, plot_args):
    """
    Plot the performance graphs.

    :param scores_dict: A dictionary holding the scores for each method on the train set
    :param scores_test_dict: A dictionary holding the scores for each method on the test set
    :param score_passive: An array holding the scores of the fully trained model
    :param path: The path to the folder where the plots will be saved
    :param scorer: The metric used to calculate the scores
    :param plot_args: A dictionary holding additional arguments

    """
    n_folds = plot_args["n_folds"]

    scores_dict_mean, scores_dict_std = get_mean_and_std(scores_dict, n_folds)
    plot_acc(scores_dict_mean, scores_dict_std, score_passive, plot_args, "{} on train set", scorer, path)

    scores_test_dict_mean, scores_test_dict_std = get_mean_and_std(scores_test_dict, n_folds)
    plot_acc(scores_test_dict_mean, scores_test_dict_std, score_passive, plot_args, "{} on test set", scorer, path)


def plot_acc(scores, stds, score_passive, plot_args, img_title="", scorer="f1_macro", path=None):
    """
    Plot the accuracy scores as a function of the queried instances.

    :param scores: Dictionary containing the accuracy scores for each method
    :param stds: Dictionary containing the standard deviations for each method
    :param score_passive: The f1 score of the experiment in a passive setting
    :param plot_args: A dictionary holding additional arguments
    :param img_title: The title of the image to be saved
    :param scorer: The metric that has been used for calculating the performance
    :param path: The path to the folder where the plots will be saved

    """
    annotated_point = plot_args["annotated_point"]
    model_name = plot_args["model_name"]
    max_iter = plot_args["max_iter"]

    # for n in range(max_iter):
    n = max_iter
    for i, (key, score) in enumerate(scores.items()):
        x = np.arange(len(score))
        plt.plot(x[:n], score[:n],
                 label=LABELS_LOOKUP.get(key, key),
                 color=COLORS[i] if i < len(COLORS) else "black",
                 marker=MARKERS[i] if i < len(MARKERS) else ".",
                 linewidth=2,
                 markevery=20)

        plt.fill_between(x[:n], score[:n] - stds[key][:n], score[:n] + stds[key][:n],
                         color=COLORS[i] if i < len(COLORS) else "black",
                         alpha=0.25,
                         linewidth=0)

        if key in annotated_point.keys():
            annotated = annotated_point[key]
            plt.annotate('start random sampling',
                         xy=(annotated, score[annotated]),
                         xytext=(annotated - 10, score[annotated] - 0.1),
                         arrowprops=dict(color="black", arrowstyle="->", connectionstyle="arc3"))

    x = np.arange(len(max(scores.values(), key=lambda value: len(value))))
    passive_mean = np.array([score_passive["mean"] for i in range(len(x))])
    passive_std = np.array([score_passive["std"] * 2 for i in range(len(x))])

    plt.plot(x[:n], passive_mean[:n], label="Passive setting", color=COLORS[-1], linewidth=2)
    plt.fill_between(x[:n], passive_mean[:n] - passive_std[:n], passive_mean[:n] + passive_std[:n],
                     alpha=0.25,
                     linewidth=0,
                     color=COLORS[-1])

    plt.xlabel('Number of obtained labels')
    plt.ylabel(scorer)
    plt.legend()
    if "banknote" in path_experiment:
        plt.ylim(0.8, 1.02)
    save_plot(plt, path, model_name, img_title.format(model_name))


def plot_narrative_bias(scores_test_dict, scores_queries_dict, plot_args, path=None):
    """
    Plot the narrative bias.

    :param scores_test_dict: A dictionary holding the scores for each method on the test set
    :param scores_queries_dict: A dictionary holding the scores calculated from the queries
    :param plot_args: A dictionary holding additional arguments
    :param path: The path to the folder where the plots will be saved

    """
    n_folds = plot_args["n_folds"]

    scores_queries_dict_mean, scores_queries_dict_std = get_mean_and_std(scores_queries_dict, n_folds)
    scores_test_dict_mean, scores_test_dict_std = get_mean_and_std(scores_test_dict, n_folds)

    for method, score in scores_queries_dict_mean.items():
        test_score = scores_test_dict_mean[method]
        x = np.arange(len(test_score))
        plt.plot(x, test_score, linewidth=2, markevery=20, label="test")
        plt.fill_between(x, test_score - scores_test_dict_std[method],
                         test_score + scores_test_dict_std[method], alpha=0.25, linewidth=0)

        x = np.arange(len(score))
        plt.plot(x, score, linewidth=1, markevery=20, label="queries", alpha=0.7)
        plt.fill_between(x, score - scores_queries_dict_std[method],
                         score + scores_queries_dict_std[method], alpha=0.25, linewidth=0)

        score_ma = running_mean(score, 10)
        x = np.arange(len(score_ma))
        plt.plot(x, score_ma, linewidth=2, markevery=20, label="queries_ma")

        plt.xlabel('Number of obtained labels')
        plt.ylabel("f1_macro")

        save_plot(plt, path, method, method)


def running_mean(data, window_width):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_width:] - cumsum[:-window_width]) / float(window_width)


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


if __name__ == '__main__':
    path_results = os.path.join(os.getcwd(), "results")
    result_folders = os.listdir(path_results)
    last_result_folder_files = os.listdir(os.path.join(path_results, result_folders[-1]))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder',
                        type=str,
                        default=result_folders[-1],
                        help="The name of the folder where the .pickle file is stored")

    args = parser.parse_args()

    pickles = [f for f in os.listdir(os.path.join(path_results, args.folder)) if f.endswith(".pickle")]
    for filename in pickles:
        experiment = filename.split("__")[0]
        path_experiment = os.path.join(path_results, args.folder, experiment)
        results = load(os.path.join(path_results, args.folder, filename))
        # Plot the results
        plot_results(results["train_f1"], results["test_f1"], results["score_passive_f1"],
                     path_experiment, "f1_macro", results["args"])
        plot_results(results["train_auc"], results["test_auc"], results["score_passive_auc"],
                     path_experiment, "roc_auc", results["args"])
        plot_narrative_bias(results["test_f1"], results["queries_f1"], results["args"], path_experiment)
