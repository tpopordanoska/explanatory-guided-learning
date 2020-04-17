import time

from src import *
from src.experiments import *

EXPERIMENTS = {
    "german": lambda **kwargs: German(**kwargs),
    "habermans-survival": lambda **kwargs: HabermansSurvival(**kwargs),
    "breast-cancer": lambda **kwargs: BreastCancer(**kwargs),
    "banknote-auth": lambda **kwargs: BanknoteAuth(**kwargs),
    "synthetic": lambda **kwargs: Synthetic(**kwargs),
    "adult": lambda **kwargs: Adult(**kwargs)
}

# Parameters for the clustering: if use_weights is true, use_labels must be true
use_labels = True
use_weights = True

# Parameters for the synthetic experiment
balanced_db = True
tiny_clusters = True

# General parameters
split_seed = 0
model_seeds = [0]
max_iter = 100
n_folds = 10
n_clusters_list = [10]
plots_off = True
# List of sampling strategies
methods = ["random", "al_least_confident", "sq_random", "rules"]
thetas = [1.0, 0.1, 0.01]
for theta in thetas:
    methods.append("xgl_{}".format(theta))

# List of experiments that will be performed
experiments = [
    "habermans-survival",
    "breast-cancer",
    "banknote-auth",
    "synthetic",
    "german",
    "adult"
]
scorers = [
    "f1_macro",
    "roc_auc"
]

# Initialization
path_to_main_file = create_folders()
for experiment_name in experiments:
    # List of models that will be run for each experiment
    experiment_path = create_experiment_folder(path_to_main_file, experiment_name)
    models = [
        # Synthetic
        # SVM(name='SVM (gamma=100, C=100)', gamma=100, C=100),
        # SVM(name='SVM (gamma=1000, C=10)', gamma=1000, C=10),

        # Banknote
        # SVM(name='SVM (gamma=1, C=10)', gamma=1, C=10),
        # NeuralNetwork()
        # GradientBoosting(),

        # Breast cancer
        # SVM(name="SVM (gamma=0.01, C=100)", gamma=0.01, C=100),
        # SVM(name="SVM"),
        # RandomForrest(),
        # NeuralNetwork(),

        # Haberman's
        # SVM(name='SVM (gamma=10, C=10)', gamma=10, C=10),
        # SVM(name="SVM (gamma=0.01, C=100)", gamma=0.01, C=100),
        # NeuralNetwork(),
        # RandomForrest(),
        # GradientBoosting(),

        # German
        # GradientBoosting(),
        # NeuralNetwork(),
        # RandomForrest()

        # Adult
        # GradientBoosting(),
        # NeuralNetwork()
        # RandomForrest(),
        # SVM(name='SVM (gamma=1, C=10)', gamma=1, C=10)
    ]
    for model in models:
        scores_dict_f1 = {}
        scores_test_dict_f1 = {}
        scores_dict_auc = {}
        scores_test_dict_auc = {}
        model_path = create_model_folder(experiment_path, model.name)

        for seed in model_seeds:
            # Set the model's seed
            model.rng = np.random.RandomState(seed)

            file = open(model_path + '\\out.txt', 'w')

            experiment = EXPERIMENTS[experiment_name](model=model, tiny_clusters=tiny_clusters, balanced_db=balanced_db)

            for n_clusters in n_clusters_list:
                learning_loop = LearningLoop(experiment, n_clusters, max_iter, model_path, file, plots_off,
                                             use_weights=use_weights, use_labels=use_labels)

                # Split the data into labeled and unlabeled
                folds = experiment.split(prop_known=experiment.prop_known, n_splits=n_folds, split_seed=split_seed)
                for k, (known_idx, train_idx, test_idx) in enumerate(folds):
                    # Remove duplicates
                    known_idx, train_idx, test_idx = np.unique(known_idx), np.unique(train_idx), np.unique(test_idx)
                    file.write('split seed {}, model seed {}, fold {} : #known {}, #train {}, #test {} \n'
                               .format(split_seed, seed, k + 1, len(known_idx), len(train_idx), len(test_idx)))
                    _, counts_known = np.unique(experiment.y[known_idx], return_counts=True)
                    _, counts_train = np.unique(experiment.y[train_idx], return_counts=True)
                    _, counts_test = np.unique(experiment.y[test_idx], return_counts=True)
                    file.write("Known bincount: {}\n".format(counts_known))
                    file.write("Train bincount: {}\n".format(counts_train))
                    file.write("Test bincount: {}\n".format(counts_test))

                    X_initial = Normalizer(experiment.normalizer).normalize(experiment.X)
                    if experiment.X.shape[1] > 2:
                        X_initial = get_tsne_embedding(X_initial)
                    plot_points(X_initial, experiment.y, "Initial points", model_path)
                    plot_points(X_initial[known_idx], experiment.y[known_idx], "Known points", model_path)
                    plot_points(X_initial[test_idx], experiment.y[test_idx], "Test points", model_path)

                    for method in methods:
                        print(method)
                        file.write("Method: {} \n".format(method))
                        file.write("Model: {}\n".format(experiment.model._model))
                        file.write("Using {} clusters, {} folds, {} model seeds, {} thetas\n".format(n_clusters, n_folds, model_seeds, thetas))
                        file.write("use_weights={}, use_labels={}, n_clusters={}\n".format(use_weights, use_labels, n_clusters))
                        start = time.time()
                        scores_f1, test_scores_f1, scores_auc, test_scores_auc = learning_loop.run(method, known_idx, train_idx, test_idx)
                        end = time.time()
                        execution_time = end-start
                        print(execution_time)
                        file.write("Execution time: " + str(execution_time))
                        # key = str(n_clusters)
                        key = method
                        if key not in scores_dict_f1:
                            scores_dict_f1[key] = [scores_f1]
                            scores_test_dict_f1[key] = [test_scores_f1]
                            scores_dict_auc[key] = [scores_auc]
                            scores_test_dict_auc[key] = [test_scores_auc]
                        else:
                            scores_dict_f1[key].append(scores_f1)
                            scores_test_dict_f1[key].append(test_scores_f1)
                            scores_dict_auc[key].append(scores_auc)
                            scores_test_dict_auc[key].append(test_scores_auc)

        plot_results(scores_dict_f1, scores_test_dict_f1, learning_loop.annotated_point, n_folds, experiment,
                     split_seed, scorers[0], file, model_path, max_iter)
        plot_results(scores_dict_auc, scores_test_dict_auc, learning_loop.annotated_point, n_folds, experiment,
                     split_seed, scorers[0], file, model_path, max_iter)
