import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from .normalizer import *
from .utils import *

sns.set()
sns.set_context("paper")


def create_meshgrid(points, h=0.1):
    """
    Create the mesh grid to plot in.

    :param h: The mesh size
    :param points: The dataset

    :return: The mesh grid to plot in
    """
    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return xx, yy


def plot_decision_surface_tsne(experiment, known_idx, train_idx, query_idx, y_pred, title, path):
    """

    Plots the decision surface of the model on TSNE-embedded data.

    :param experiment: The experiment
    :param known_idx: The indexes of the known points
    :param train_idx: The indexes of the training points
    :param query_idx: The index of chosen least confident example
    :param y_pred: The predictions of the model
    :param title: The title of the plot
    :param path: The path of the folder where the plot will be saved

    """
    X_known, y_known = get_from_indexes(experiment.X, known_idx), experiment.y[known_idx]
    X_train, y_train = get_from_indexes(experiment.X, train_idx), experiment.y[train_idx]
    X_known_norm, X_train_norm = Normalizer(experiment.normalizer).normalize_known_train(X_known, X_train)
    X_known_train_norm = np.concatenate((X_known_norm, X_train_norm), axis=0)

    X_initial = Normalizer(experiment.normalizer).normalize(experiment.X)
    X_embedded = get_tsne_embedding(X_initial)
    X_known_embedded = X_embedded[known_idx]
    X_train_embedded = X_embedded[train_idx]
    X_known_train_embedded = np.concatenate((X_known_embedded, X_train_embedded), axis=0)

    y_predicted = experiment.model.predict(X_known_train_norm)

    h = 10
    x_min, x_max = X_known_train_embedded[:, 0].min() - 1, X_known_train_embedded[:, 0].max() + 1
    y_min, y_max = X_known_train_embedded[:, 1].min() - 1, X_known_train_embedded[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_known_train_embedded, y_predicted)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((h, h))

    figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
    plt.contourf(xx, yy, voronoiBackground, cmap=plt.cm.RdBu_r, alpha=0.8)

    if y_pred is not None:
        plt.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], c=y_pred, cmap=plt.cm.RdBu_r, s=45)
    else:
        plt.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], c=y_train, cmap=plt.cm.RdBu_r, s=45)
        plt.scatter(X_known_embedded[:, 0], X_known_embedded[:, 1], c=y_known, cmap=plt.cm.RdBu_r, s=45,
                    edgecolors="yellow", linewidths=2)

    plt.title(title)
    if query_idx is not None:
        least_conf = get_from_indexes(experiment.X, query_idx)
        if isinstance(experiment, Adult):
            X_train = X_train.to_numpy()
        idx_array = np.where((X_train[:, 0] == least_conf[0]) & (X_train[:, 1] == least_conf[1]))[0]
        if len(idx_array):
            idx_in_train = idx_array[0]
            least_conf_norm = get_from_indexes(X_train_embedded, idx_in_train)
            plt.scatter(least_conf_norm[0], least_conf_norm[1], marker='x', s=400, linewidths=5, color='yellow')

    if path:
        plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + "-" + title + '.png')
    else:
        plt.show()
    plt.close()


def plot_decision_surface(experiment, known_idx, train_idx, query_idx=None, y_pred=None, soft=True, title="",
                          path=None):
    """
    Plots the decision surface of the model together with the data points.

    :param experiment: The experiment
    :param known_idx: The indexes of the known points
    :param train_idx: The indexes of the training points
    :param query_idx: The index of chosen least confident example
    :param y_pred: The predictions of the model
    :param soft: Whether to plot  kernel-like boundary
    :param title: The title of the plot
    :param path: The path of the folder where the plot will be saved

    """
    if experiment.X.shape[1] > 2:
        plot_decision_surface_tsne(experiment, known_idx, train_idx, query_idx, y_pred, title, path)
    else:
        X_known, y_known = experiment.X[known_idx], experiment.y[known_idx]
        X_train, y_train = experiment.X[train_idx], experiment.y[train_idx]
        X_known_norm, X_train_norm = Normalizer(experiment.normalizer).normalize_known_train(X_known, X_train)
        X_known_train_norm = np.concatenate((X_known_norm, X_train_norm), axis=0)
        model = experiment.model

        figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
        # create a mesh to plot in
        h = 0.05  # step size in the mesh
        xx, yy = create_meshgrid(X_known_train_norm, h)

        if soft:
            if hasattr(model._model, "decision_function"):
                Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.8)

        if y_pred is not None:
            wrong_points = X_train_norm[np.where(y_pred != y_train)]
            plt.scatter(X_train_norm[:, 0], X_train_norm[:, 1], c=y_pred, cmap=plt.cm.RdBu_r, s=45)
            plt.scatter(wrong_points[:, 0], wrong_points[:, 1], s=120, cmap=plt.cm.RdBu_r, facecolors='none',
                        edgecolors='green', linewidths=2)
        else:
            plt.scatter(X_train_norm[:, 0], X_train_norm[:, 1], c=y_train, cmap=plt.cm.RdBu_r, s=45)
            plt.scatter(X_known_norm[:, 0], X_known_norm[:, 1], c=y_known, cmap=plt.cm.RdBu_r, s=45,
                    edgecolors="yellow", linewidths=2)

        if query_idx is not None:
            least_conf = experiment.X[query_idx]
            idx_array = np.where((X_train[:, 0] == least_conf[0]) & (X_train[:, 1] == least_conf[1]))[0]
            if len(idx_array):
                idx_in_train = idx_array[0]
                least_conf_norm = X_train_norm[idx_in_train]
                plt.scatter(least_conf_norm[0], least_conf_norm[1], marker='x', s=400, linewidths=5, color='yellow')
        plt.title(title)
        plt.xlim(X_known_train_norm[:, 0].min() - 0.1, X_known_train_norm[:, 0].max() + 0.1)
        plt.ylim(X_known_train_norm[:, 1].min() - 0.1, X_known_train_norm[:, 1].max() + 0.1)
        if path:
            plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + "-" + title + '.png',
                        bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def plot_points(X, y, title="", path=None):
    """
    Plot the given points with their corresponding labels.

    :param X: Contains the coordinates of the points to be plotted
    :param y: The corresponding labels
    :param title: The title of the plot
    :param path: The path of the folder where the plot will be saved

    """
    figure(num=None, figsize=(10, 8), facecolor='w', edgecolor='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=45, cmap=plt.cm.coolwarm)
    # set axes range
    plt.xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    plt.ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    # plt.title(title)
    # plt.axis("off")
    if path:
        plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + "-" + title + '.png',
                    bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def get_tsne_embedding(X):
    return TSNE(n_components=2, n_iter=300, random_state=0).fit_transform(X)


def plot_results(scores_dict, scores_test_dict, annotated_point, n_folds, experiment,
                 split_seed, scorer, file, path, max_iter):
    """
    Plot the performance graphs.

    :param scores_dict: A dictionary holding the scores for each method on the train set
    :param scores_test_dict: A dictionary holding the scores for each method on the test set
    :param annotated_point: The point where random sampling starts in XGL.
    :param n_folds: The number of folds
    :param experiment: The experiment
    :param split_seed: The seed used for the split
    :param scorer: The metric
    :param file: The output file
    :param path: The path of the folder where the plot will be saved
    :param max_iter: The maximal iteration, used for the

    """

    score_passive = get_passive_score(experiment, file, n_folds, split_seed, scorer)

    scores_dict_mean, scores_dict_std = get_mean_and_std(scores_dict, n_folds)
    plot_acc(scores_dict_mean, scores_dict_std, score_passive,
             annotated_dict=annotated_point,
             plot_title=experiment.model.name,
             img_title="{} on train set".format(experiment.model.name),
             scorer=scorer,
             path=path,
             max_iter=max_iter)

    scores_test_dict_mean, scores_test_dict_std = get_mean_and_std(scores_test_dict, n_folds)
    plot_acc(scores_test_dict_mean, scores_test_dict_std, score_passive,
             annotated_dict=annotated_point,
             plot_title=experiment.model.name,
             img_title="{} on test set".format(experiment.model.name),
             scorer=scorer,
             path=path,
             max_iter=max_iter)


def plot_acc(scores, stds, score_passive, annotated_dict=None, img_title="", plot_title="", path=None,
             scorer="f1_weighted", max_iter="100"):
    """
    Plot the accuracy scores as a function of the queried instances.

    :param scores: Dictionary containing the accuracy scores for each method
    :param stds: Dictionary containing the standard deviations for each method
    :param score_passive: The f1 score of the experiment in a passive setting
    :param annotated_dict: Dictionary containing the point where we switch to random sampling in XGL
    :param img_title: The title of the image saved
    :param plot_title: The title of the plot
    :param path: The path of the folder where the plot will be saved
    :param scorer: The metric that has been used for calculating the performance
    :param max_iter: The maximal iteration, used for the plots for GIFs

    """
    # colors = ['#840000', 'red', '#fc824a']
    # linestyle=['solid', 'dotted', 'dashed']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown']
    markers = ['o', 'v', 's', 'D', 'p', '*', 'd', '<', '<']
    labels_lookup = {
        "random": "Random sampling",
        "al_least_confident": "Active Learning",
        "sq_random": "Guided Learning",
        "xgl_100.0": "XGL (theta=100)",
        "xgl_5": "XGL (theta=5)",
        "xgl_10.0": "XGL (theta=10)",
        "xgl_1.0": "XGL (theta=1)",
        "xgl_0.1": "XGL (theta=0.1)",
        "xgl_0.01": "XGL (theta=0.01)",
        "10": "10 prototypes",
        "30": "30 prototypes",
        "50": "50 prototypes",
        "100": "100 prototypes"
    }
    # for n in range(max_iter):
    n = max_iter
    for i, (key, score) in enumerate(scores.items()):
        x = np.arange(len(score))
        plt.plot(x[:n], score[:n], label=labels_lookup[key], color=colors[i], linewidth=2, marker=markers[i], markevery=20)
        plt.fill_between(x[:n], score[:n] - stds[key][:n], score[:n] + stds[key][:n], alpha=0.25, linewidth=0, color=colors[i])
        if key in annotated_dict.keys():
            annotated = annotated_dict[key]
            plt.annotate('start random sampling', xy=(annotated, score[annotated]), xytext=(annotated - 100, score[annotated] - 0.1),
                         arrowprops=dict(color="black", arrowstyle="->", connectionstyle="arc3"))

    x = np.arange(len(max(scores.values(), key=lambda value: len(value))))
    passive_mean = np.array([score_passive["mean"] for i in range(len(x))])
    passive_std = np.array([score_passive["std"] * 2 for i in range(len(x))])

    plt.plot(x[:n], passive_mean[:n], label="Passive setting", color=colors[-1], linewidth=2)
    plt.fill_between(x[:n], passive_mean[:n] - passive_std[:n], passive_mean[:n] + passive_std[:n], alpha=0.25, linewidth=0,
                     color=colors[-1])

    plt.grid(True)
    plt.xlabel('Number of obtained labels')
    plt.ylabel(scorer)
    plt.title(plot_title)
    plt.legend()
    if path:
        try:
            plt.savefig(path + "\\" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f') + "-" + img_title + '.png',  bbox_inches='tight')
        except ValueError:
            print("Something wrong with plotting")

    else:
        plt.show()
    plt.close()
