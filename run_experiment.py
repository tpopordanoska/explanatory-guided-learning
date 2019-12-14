from src import *
from src.experiments import *
from sklearn.model_selection import cross_val_score

EXPERIMENTS = {
    "habermans-survival": lambda **kwargs: HabermansSurvival(**kwargs),
    "breast-cancer": lambda **kwargs: BreastCancer(**kwargs),
    "banknote-auth": lambda **kwargs: BanknoteAuth(**kwargs),
    "synthetic-simple": lambda **kwargs: SyntheticSimple(**kwargs),
    "synthetic": lambda **kwargs: Synthetic(**kwargs),
    "adult": lambda **kwargs: Adult(**kwargs)
}


def get_mean_and_std(scores_dict):
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


def get_passive_f1(experiment, file, n_splits, rng):
    exp_model = experiment.model._model
    X_normalized = Normalizer(experiment.normalizer).normalize(experiment.X)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    scores = cross_val_score(exp_model, X_normalized, experiment.y, cv=kfold, scoring="f1")
    file.write("Passive accuracy for {}: {:.2f} (+/- {:.2f}) ".format(experiment.name, scores.mean(), scores.std() * 2))
    return scores

# Parameters for the clustering: if use_weights is true, use_labels must be true
use_labels = True
use_weights = True

# Parameters for the synthetic experiment
balanced_db = True
tiny_clusters = True

# Initialization
path_to_main_file = create_folders()

# General parameters
seeds=[0]
# seeds = [0, 25, 42, 64, 100]
max_iter = 100
n_folds = 3
n_clusters = 10
plots_off = False
# List of sampling strategies
methods = ["al_least_confident", "sq_random"]
thetas = [1, 0.1, 0.01]
for theta in thetas:
    methods.append("optimal_user_{}".format(theta))
# List of experiments that will be performed
experiments = [
    "habermans-survival",
    "breast-cancer",
    "banknote-auth",
    "synthetic",
    "synthetic-simple",
    "adult"
]

for experiment_name in experiments:
    # TODO: fix this to average over different seeds
    for seed in seeds:
        rng = np.random.RandomState(seed)
        # List of models that will be run for each experiment
        models = [
            SVM(name="default"),
            GNB(name='gaussian-naive-bayes', rng=rng),
            SVM(name='svm_rbf', rng=rng, gamma=1e3, C=1),
            SVM(name='svm_rbf', rng=rng, gamma=1, C=1e2),
            NeuralNetwork(name="MLP", rng=rng)
        ]
        for model in models:
            scores_dict = {}
            scores_test_dict = {}

            results_path = create_experiment_folder(path_to_main_file, experiment_name, model.name)
            file = open(results_path + '\\out.txt', 'w')

            experiment = EXPERIMENTS[experiment_name](model=model,
                                                      rng=rng,
                                                      tiny_clusters=tiny_clusters,
                                                      balanced_db=balanced_db)

            X_normalized = Normalizer(experiment.normalizer).normalize(experiment.X)
            plot_points(X_normalized, experiment.y, "Initial points", results_path)

            learning_loop = ActiveLearningLoop(experiment, n_clusters, max_iter, results_path, file, plots_off, thetas,
                                               use_weights=use_weights, use_labels=use_labels)

            # Split the data into labeled and unlabeled
            folds = experiment.split(prop_known=experiment.prop_known, n_splits=n_folds)
            for k, (known_idx, train_idx, test_idx) in enumerate(folds):
                # Remove duplicates
                known_idx, train_idx, test_idx = np.unique(known_idx), np.unique(train_idx), np.unique(test_idx)
                file.write('fold {} : #known {}, #train {}, #test {} \n'
                           .format(k + 1, len(known_idx), len(train_idx), len(test_idx)))
                _, counts_known = np.unique(experiment.y[known_idx], return_counts=True)
                _, counts_train = np.unique(experiment.y[train_idx], return_counts=True)
                _, counts_test = np.unique(experiment.y[test_idx], return_counts=True)
                file.write("Known bincount: {}\n".format(counts_known))
                file.write("Train bincount: {}\n".format(counts_train))
                file.write("Test bincount: {}\n".format(counts_test))

                plot_points(X_normalized[known_idx], experiment.y[known_idx], "Known points", results_path)
                plot_points(X_normalized[test_idx], experiment.y[test_idx], "Test points", results_path)

                for method in methods:
                    print(method)
                    file.write("Method: {} \n".format(method))
                    file.write("Model: {}\n".format(experiment.model._model))
                    file.write("Using {} clusters, {} folds, {} seeds, {} thetas\n".format(n_clusters, folds, seeds, thetas))
                    file.write("use_weights={}, use_labels={}\n".format(use_weights, use_labels))
                    acc_scores, test_acc_scores = learning_loop.run(method, known_idx, train_idx, test_idx)

                    if method not in scores_dict:
                        scores_dict[method] = [acc_scores]
                        scores_test_dict[method] = [test_acc_scores]
                    else:
                        scores_dict[method].append(acc_scores)
                        scores_test_dict[method].append(test_acc_scores)

            scores_dict_mean, scores_dict_std = get_mean_and_std(scores_dict)
            scores_test_dict_mean, scores_test_dict_std = get_mean_and_std(scores_test_dict)

            f1_score_passive = get_passive_f1(experiment, file, n_folds, rng)
            file.write("Passive accuracy for {}:  {:.2f} (+/- {:.2f}) "
                       .format(experiment.name, f1_score_passive.mean(), f1_score_passive.std() * 2))

            plot_acc(scores_dict_mean, scores_dict_std, f1_score_passive, title="{} {} F1 score on train set using {}"
                     .format(experiment.model.name, experiment.name, str(n_clusters)), path=results_path)

            plot_acc(scores_test_dict_mean, scores_test_dict_std, f1_score_passive, title="{} {} F1 score on test set using {}"
                     .format(experiment.model.name, experiment.name, str(n_clusters)), path=results_path)
