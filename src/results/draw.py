import matplotlib.pyplot as plt

from src.utils.arguments import get_drawing_args
from src.utils.common import *
from src.utils.constants import *
from src.utils.plotting import save_plot

FORMATTING_METHODS = {
    "al_dw_1": "al-dw",
    "al_lc": "al-us",
    "xgl_rules_100.0": "xgl-rules",
    "sq_random": "gl",
    "xgl_rules_10.0": "xgl-rules10",
    "xgl_rules_1.0": "xgl-rules1",
    "xgl_rules_hierarchy_100.0": "xgl-rules_hierarchy",
    "xgl_rules_hierarchy_10.0": "rules_hierarchy",
    "xgl_rules_hierarchy_1.0":  "xgl-rules_hierarchy",
    "xgl_clusters_1.0": "xgl",
}

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_results(scores_dict, scores_test_dict, score_passive, path, scorer, plot_args, strategies, exp, title=""):
    """
    Plot the performance graphs.

    :param scores_dict: A dictionary holding the scores for each method on the train set
    :param scores_test_dict: A dictionary holding the scores for each method on the test set
    :param score_passive: An array holding the scores of the fully trained model
    :param path: The path to the folder where the plots will be saved
    :param scorer: The metric used to calculate the scores
    :param plot_args: A dictionary holding additional arguments
    :param strategies: A list of strategies to be plotted
    :param title: Extra information to be added in the title

    """
    n_folds = plot_args["n_folds"]

    scores_dict_mean, scores_dict_std = get_mean_and_std(scores_dict, n_folds)
    plot_acc(scores_dict_mean, scores_dict_std, score_passive, plot_args, strategies, " train set", scorer, path)

    scores_test_dict_mean, scores_test_dict_std = get_mean_and_std(scores_test_dict, n_folds)
    plot_acc(scores_test_dict_mean, scores_test_dict_std, score_passive,
             plot_args, strategies, exp + title, scorer, path)


def plot_acc(scores, stds, score_passive, plot_args, strategies, img_title="", scorer="f1_macro", path=None):
    """
    Plot the accuracy scores as a function of the queried instances.

    :param scores: Dictionary containing the accuracy scores for each method
    :param stds: Dictionary containing the standard deviations for each method
    :param score_passive: The f1 score of the experiment in a passive setting
    :param plot_args: A dictionary holding additional arguments
    :param strategies: A list of strategies to be plotted
    :param img_title: The title of the image to be saved
    :param scorer: The metric that has been used for calculating the performance
    :param path: The path to the folder where the plots will be saved

    """
    annotated_point = plot_args["annotated_point"]
    model_name = plot_args["model_name"]
    max_iter = plot_args["max_iter"]

    # for n in range(max_iter):
    n = max_iter
    i = 0
    for key, score in sorted(scores.items()):
        if key not in strategies:
            continue
        param, method = extract_param_from_name(key)
        x = np.arange(len(score))
        plt.plot(x[:n], score[:n],
                 label=format_label(method, param),
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
        i += 1

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
    save_plot(plt, path, img_title, model_name, use_date=False)


def plot_narrative_bias(scores_test_dict, scores_queries_dict, plot_args, exp, path=None):
    """
    Plot the narrative bias.

    :param scores_test_dict: A dictionary holding the scores for each method on the test set
    :param scores_queries_dict: A dictionary holding the scores calculated from the queries
    :param plot_args: A dictionary holding additional arguments
    :param exp: The experiment name
    :param path: The path to the folder where the plots will be saved

    """
    n_folds = plot_args["n_folds"]

    scores_queries_dict_mean, scores_queries_dict_std = get_mean_and_std(scores_queries_dict, n_folds)
    scores_test_dict_mean, scores_test_dict_std = get_mean_and_std(scores_test_dict, n_folds)

    for method, score in sorted(scores_queries_dict_mean.items()):
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
        plt.legend()
        img_title = "nb_{}_{}".format(exp, FORMATTING_METHODS.get(method, method))
        save_plot(plt, path, method, img_title, use_date=False)


def plot_grouped_narrative_bias_weighted(scores_test_dict, scores_queries_dict, plot_args, exp, arg_strategies, path):
    plot_grouped_narrative_bias(scores_test_dict, scores_queries_dict, plot_args,
                                exp, arg_strategies, path, " - weighted")


def plot_grouped_narrative_bias(scores_test_dict,
                                scores_queries_dict,
                                plot_args,
                                exp,
                                arg_strategies,
                                path=None,
                                title=""):

    n_folds = plot_args["n_folds"]
    i = 0
    for method, scores_queries in sorted(scores_queries_dict.items()):
        if method not in arg_strategies:
            continue
        scores_test = scores_test_dict[method]
        differences = []
        for score_queries, score_test in zip(scores_queries, scores_test):
            score_ma = running_mean(score_queries, 20)
            differences.append(score_ma - score_test[:len(score_ma)])

        smallest_len = min([len(x) for x in differences])
        differences_smallest_len = [s[:smallest_len] for s in differences]
        difference_mean = np.mean(differences_smallest_len, axis=0)
        difference_std = np.std(differences_smallest_len, axis=0) / np.sqrt(n_folds)

        x = np.arange(len(difference_mean))
        param, key = extract_param_from_name(method)
        plt.plot(x, difference_mean,
                 color=COLORS[i] if i < len(COLORS) else "black",
                 label=format_label(key, param),
                 marker=MARKERS[i] if i < len(MARKERS) else ".",
                 linewidth=2,
                 markevery=20)

        plt.fill_between(x, difference_mean - difference_std, difference_mean + difference_std,
                         color=COLORS[i] if i < len(COLORS) else "black",
                         alpha=0.25,
                         linewidth=0)
        plt.xlabel('Number of obtained labels')
        plt.legend()
        i += 1

    nb_path = create_folder(path, "narrative_bias")
    save_plot(plt, nb_path, exp + title, f"Narrative bias{title}", use_date=False)


def plot_false_mistakes(false_mistakes_dict, path, strategies):
    for strategy, false_mistakes in sorted(false_mistakes_dict.items()):
        if strategy not in strategies or "rules" not in strategy:
            continue
        smallest_len = min([len(x) for x in false_mistakes])
        false_mistakes_smallest_len = [s[:smallest_len] for s in false_mistakes]
        false_mistakes_mean = np.mean(false_mistakes_smallest_len, axis=0)
        x = np.arange(len(false_mistakes_mean))
        param, method = extract_param_from_name(strategy)
        plt.plot(x, false_mistakes_mean, linewidth=2, markevery=20, label=format_label(method, param))

        plt.xlabel('Number of obtained labels')
        plt.ylabel("Number of false mistakes")
        plt.legend()

    save_plot(plt, path, "False mistakes", "False mistakes", use_date=False)


def plot_rules_wrt_blackbox_f1(rules_f1_dict, plot_args, path):
    n_folds = plot_args["n_folds"]
    scores_rules_dict_mean, scores_rules_dict_std = get_mean_and_std(rules_f1_dict, n_folds)

    for method, score in sorted(scores_rules_dict_mean.items()):
        x = np.arange(len(score))
        plt.plot(x, score, linewidth=2, markevery=20, label=method)
        plt.fill_between(x, score - scores_rules_dict_std[method],
                         score + scores_rules_dict_std[method], alpha=0.25, linewidth=0)
        plt.legend()

    save_plot(plt, path, "Rules f1 wrt blackbox", "Rules f1 wrt blackbox", use_date=False)


def running_mean(data, window_width):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_width:] - cumsum[:-window_width]) / float(window_width)


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def format_label(method, param):
    if method == 'random':
        return "Random"
    elif method == 'al_lc':
        return "AL (unc.)"
    elif method == 'al_dw':
        return "AL (repr.)"
    elif method == 'sq_random':
        return "GL"
    elif method == 'xgl_clusters':
        return "XGL(clustering) (θ={})".format(param)
    elif method == 'xgl_rules_hierarchy':
        return "XGL (hier.) (θ={})".format(param)
    elif method == 'xgl_rules':
        return "XGL (θ={})".format(param)
    else:
        return method


if __name__ == '__main__':
    path_results = os.path.join(os.getcwd(), "results")
    result_folders = os.listdir(path_results)

    args = get_drawing_args(result_folders)
    strategies = create_strategies_list(args)
    path_folder = os.path.join(path_results, args.folder)

    pickles = [f for f in os.listdir(path_folder) if f.endswith(".pickle")]
    for filename in pickles:
        experiment = filename.split("__")[0]
        path_experiment = create_folder((os.path.join(path_results, args.folder)), experiment)
        results = load(os.path.join(path_results, args.folder, filename))
        # Plot the results
        plot_results(results["train_f1"], results["test_f1"], results["score_passive_f1"],
                     path_experiment, "avg $F_1$", results["args"], strategies, experiment)

        plot_results(results["train_f1_weighted"], results["test_f1_weighted"], results["score_passive_f1"],
                     path_experiment, "avg $F_1$", results["args"], strategies, experiment, title="- weighted")

        plot_narrative_bias(results["test_f1"], results["queries_f1"], results["args"], experiment, path_experiment)

        plot_grouped_narrative_bias(results["test_f1"], results["queries_f1"], results["args"],
                                    experiment, strategies, path_folder)

        plot_grouped_narrative_bias_weighted(results["test_f1_weighted"], results["queries_f1"], results["args"],
                                     experiment, strategies, path_folder)

        plot_false_mistakes(results["false_mistakes"], path_experiment, strategies)

        plot_rules_wrt_blackbox_f1(results["rules_wrt_blackbox_f1"], results['args'], path_experiment)


