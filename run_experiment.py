import time
from collections import defaultdict

from src import *
from src.experiments import *

EXPERIMENTS = {
    "german": lambda **kwargs: German(**kwargs),
    "habermans-survival": lambda **kwargs: HabermansSurvival(**kwargs),
    "breast-cancer": lambda **kwargs: BreastCancer(**kwargs),
    "banknote-auth": lambda **kwargs: BanknoteAuth(**kwargs),
    "synthetic": lambda **kwargs: Synthetic(**kwargs),
    "adult": lambda **kwargs: Adult(**kwargs),
    "credit": lambda **kwargs: Credit(**kwargs),
    "australian": lambda **kwargs: Australian(**kwargs),
    "hepatitis": lambda **kwargs: Hepatitis(**kwargs),
    "heart": lambda **kwargs: Heart(**kwargs)
}

# Parameters for the clustering: if use_weights is true, use_labels must be true
use_labels = True
use_weights = True

# Parameters for the synthetic experiment
balanced_db = True
tiny_clusters = True

# General parameters
split_seed = 0
seed = 0
max_iter = 20
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
    # "habermans-survival",
    # "breast-cancer",
    # "banknote-auth",
    "synthetic",
    # "german",
    # "adult",
    # "credit",
    # "australian",
    # "hepatitis",
    # "heart"
]
scorers = [
    "f1_macro",
    "roc_auc"
]

# Initialization
path_to_main_file = create_folders()
for experiment_name in experiments:
    # Initialize the experiment
    experiment = EXPERIMENTS[experiment_name](rng=np.random.RandomState(seed),
                                              tiny_clusters=tiny_clusters,
                                              balanced_db=balanced_db)

    # Create folders and file for logging
    experiment_path = create_experiment_folder(path_to_main_file, experiment_name)
    model_path = create_model_folder(experiment_path, experiment.model.name)
    file = open(model_path + '\\out.txt', 'w')

    # Initialize results dictionaries
    scores_dict_f1 = defaultdict(list)
    scores_test_dict_f1 = defaultdict(list)
    scores_dict_auc = defaultdict(list)
    scores_test_dict_auc = defaultdict(list)
    scores_queries_dict_f1 = defaultdict(list)

    for n_clusters in n_clusters_list:
        # Initialize the learning loop
        loop = LearningLoop(experiment, n_clusters, max_iter, model_path, file, plots_off, use_weights, use_labels)

        # Split the data into initially labeled (known), pool of unlabeled to choose from (train) and unlabeled for test
        folds = experiment.split(prop_known=experiment.prop_known, n_splits=n_folds, split_seed=split_seed)
        for k, (known_idx, train_idx, test_idx) in enumerate(folds):
            # Remove duplicates
            known_idx, train_idx, test_idx = np.unique(known_idx), np.unique(train_idx), np.unique(test_idx)

            # Plot the points
            if not plots_off:
                X_initial = Normalizer(experiment.normalizer).normalize(experiment.X)
                if experiment.X.shape[1] > 2:
                    X_initial = get_tsne_embedding(X_initial)
                plot_points(X_initial, experiment.y, "Initial points", model_path)
                plot_points(X_initial[known_idx], experiment.y[known_idx], "Known points", model_path)
                plot_points(X_initial[test_idx], experiment.y[test_idx], "Test points", model_path)

            # Run the experiment for every method in the list
            for method in methods:
                # Write the parameters to the output file
                write_to_file(file, known_idx, train_idx, test_idx, seed, k, experiment,
                              method, n_clusters, n_folds, thetas, use_weights, use_labels)

                # Run the experiment
                start = time.time()
                loop.run(method, known_idx, train_idx, test_idx)
                end = time.time()
                execution_time = end-start
                file.write("Execution time: " + str(execution_time))

                # Collect the results for each method in a dictionary
                key = method  # Use key = str(n_clusters) for checking the effect of n_clusters
                scores_dict_f1[key].append(loop.scores_f1)
                scores_test_dict_f1[key].append(loop.test_scores_f1)
                scores_dict_auc[key].append(loop.scores_auc)
                scores_test_dict_auc[key].append(loop.test_scores_auc)

        # Plot results
        plot_results(scores_dict_f1, scores_test_dict_f1, loop.annotated_point, n_folds, experiment,
                     split_seed, scorers[0], file, model_path, max_iter)
        plot_results(scores_dict_auc, scores_test_dict_auc, loop.annotated_point, n_folds, experiment,
                     split_seed, scorers[1], file, model_path, max_iter)
