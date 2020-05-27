import time
from collections import defaultdict

from constants import EXPERIMENTS
from src import *

# Parameters for the clustering: if use_weights is true, use_labels must be true
use_labels = True
use_weights = True

# Parameters for the synthetic experiment
balanced_db = True
tiny_clusters = True

# General parameters
split_seed = 0
seed = 0
max_iter = 100
n_folds = 10
n_clusters_list = [10]
plots_off = True
# List of sampling strategies
methods = ["random", "al_least_confident", "sq_random"]
betas = [1]
thetas_xgl = [1.0]
thetas_rules = [100.0]
for theta_xgl in thetas_xgl:
    methods.append("xgl_{}".format(theta_xgl))
for theta_rules in thetas_rules:
    methods.append("rules_{}".format(theta_rules))
    methods.append("rules_hierarchy_{}".format(theta_rules))
for beta in betas:
    methods.append("al_density_weighted_{}".format(beta))

# List of experiments that will be performed
experiments = [
    "breast-cancer",
    "banknote-auth",
    "synthetic",
    "german",
    "adult",
    "credit",
    "australian",
    "hepatitis",
    "heart"
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
    file = open(os.path.join(experiment_path, 'out.txt'), 'w')

    # Initialize results dictionaries
    scores_dict_f1 = defaultdict(list)
    scores_test_dict_f1 = defaultdict(list)
    scores_dict_auc = defaultdict(list)
    scores_test_dict_auc = defaultdict(list)
    scores_queries_dict_f1 = defaultdict(list)

    for n_clusters in n_clusters_list:
        # Initialize the learning loop
        loop = LearningLoop(experiment, n_clusters, max_iter, experiment_path, file, plots_off, use_weights, use_labels)

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
                plot_points(X_initial, experiment.y, "Initial points", experiment_path)
                plot_points(X_initial[known_idx], experiment.y[known_idx], "Known points", experiment_path)
                plot_points(X_initial[test_idx], experiment.y[test_idx], "Test points", experiment_path)

            # Run the experiment for every method in the list
            for method in methods:
                # Write the parameters to the output file
                write_to_file(file, known_idx, train_idx, test_idx, seed, k, experiment,
                              method, n_clusters, n_folds, thetas_xgl, use_weights, use_labels)

                # Run the experiment
                start = time.time()
                loop.run(method, known_idx, train_idx, test_idx)
                end = time.time()
                file.write("Execution time: " + str(end-start))

                # Collect the results for each method in a dictionary
                key = method  # Use key = str(n_clusters) for checking the effect of n_clusters
                scores_dict_f1[key].append(loop.scores_f1)
                scores_test_dict_f1[key].append(loop.test_scores_f1)
                scores_dict_auc[key].append(loop.scores_auc)
                scores_test_dict_auc[key].append(loop.test_scores_auc)
                scores_queries_dict_f1[key].append(loop.query_scores)

    dump(experiment_path + '__scores.pickle', {
        'train_f1': scores_dict_f1,
        'test_f1': scores_test_dict_f1,
        'train_auc': scores_dict_auc,
        'test_auc': scores_test_dict_auc,
        'queries_f1': scores_queries_dict_f1,
        'score_passive_f1': get_passive_score(experiment, file, n_folds, split_seed, "f1_macro"),
        'score_passive_auc': get_passive_score(experiment, file, n_folds, split_seed, "roc_auc"),
        'args': {
            'n_folds': n_folds,
            'max_iter': max_iter,
            'model_name': experiment.model.name,
            'annotated_point': loop.annotated_point,
        },
    })

