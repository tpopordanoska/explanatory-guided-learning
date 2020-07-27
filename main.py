import time
import warnings
from collections import defaultdict

from arguments import get_main_args
from src import *

warnings.filterwarnings('ignore')

# Parameters for the clustering: if use_weights is true, use_labels must be true
use_labels = True
use_weights = True


def run_experiment(strategies, args):
    path_to_main_file = create_folders()
    for experiment_name in args.experiments:
        experiment, experiment_path, file = initialize(experiment_name, path_to_main_file, args.seed)

        # Initialize results dictionaries
        scores_dict_f1 = defaultdict(list)
        scores_test_dict_f1 = defaultdict(list)
        scores_dict_auc = defaultdict(list)
        scores_test_dict_auc = defaultdict(list)
        scores_queries_dict_f1 = defaultdict(list)
        false_mistakes_dict = defaultdict(list)
        check_rules_dict = defaultdict(list)

        # Initialize the learning loop
        loop = LearningLoop(experiment, args.n_clusters, args.max_iter, experiment_path,
                            file, args.plots_on, use_weights, use_labels)

        # Introduce unknown unknowns by flipping the class of random sub-groups from the training data
        introduce_uu(experiment)

        # Split the data into initially labeled (known), pool of unlabeled to choose from (train) and unlabeled for test
        folds = experiment.split(prop_known=experiment.prop_known, n_splits=args.n_folds, split_seed=args.seed)
        for k, (known_idx, train_idx, test_idx) in enumerate(folds):
            # Remove duplicates
            known_idx, train_idx, test_idx = np.unique(known_idx), np.unique(train_idx), np.unique(test_idx)

            # Plot the points
            if args.plots_on:
                X_initial = Normalizer(experiment.normalizer).normalize(experiment.X)
                if experiment.X.shape[1] > 2:
                    X_initial = get_tsne_embedding(X_initial)
                plot_points(X_initial, experiment.y, "Initial points", experiment_path)
                plot_points(X_initial[known_idx], experiment.y[known_idx], "Known points", experiment_path)
                plot_points(X_initial[test_idx], experiment.y[test_idx], "Test points", experiment_path)

            # Run the experiment for every strategy
            for strategy in strategies:
                # Write the parameters to the output file
                write_to_file(file, known_idx, train_idx, test_idx, args.seed, k, experiment,
                              strategy, args.n_clusters, args.n_folds, use_weights, use_labels)

                # Run the experiment
                start = time.time()
                loop.run(strategy, known_idx, train_idx, test_idx)
                end = time.time()
                file.write("Execution time: {} \n".format(str(end - start)))

                # Collect the results for each strategy in a dictionary
                scores_dict_f1[strategy].append(loop.scores_f1)
                scores_test_dict_f1[strategy].append(loop.test_scores_f1)
                scores_dict_auc[strategy].append(loop.scores_auc)
                scores_test_dict_auc[strategy].append(loop.test_scores_auc)
                scores_queries_dict_f1[strategy].append(loop.query_scores)
                false_mistakes_dict[strategy].append(loop.false_mistakes_count)
                check_rules_dict[strategy].append(loop.check_rules_f1)

        dump(experiment_path + '__scores.pickle', {
            'train_f1': scores_dict_f1,
            'test_f1': scores_test_dict_f1,
            'train_auc': scores_dict_auc,
            'test_auc': scores_test_dict_auc,
            'queries_f1': scores_queries_dict_f1,
            'check_rules_f1': check_rules_dict,
            'score_passive_f1': get_passive_score(experiment, file, args.n_folds, args.seed, "f1_macro"),
            'score_passive_auc': get_passive_score(experiment, file, args.n_folds, args.seed, "roc_auc"),
            'false_mistakes': false_mistakes_dict,
            'args': {
                'n_folds': args.n_folds,
                'max_iter': args.max_iter,
                'model_name': experiment.model.name,
                'annotated_point': loop.annotated_point,
            },
        })


def main():
    args = get_main_args()
    strategies = create_strategies_list(args)

    np.random.seed(args.seed)
    run_experiment(strategies, args)


if __name__ == '__main__':
    main()
