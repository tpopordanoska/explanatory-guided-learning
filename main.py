import time
import warnings
from collections import defaultdict

from constants import EXPERIMENTS, STRATEGIES
from src import *

warnings.filterwarnings('ignore')

# Parameters for the clustering: if use_weights is true, use_labels must be true
use_labels = True
use_weights = True

# Parameters for the synthetic experiment
balanced_db = True
tiny_clusters = True


def run_experiment(strategies, args):
    # Initialization
    path_to_main_file = create_folders()
    for experiment_name in args.experiments:
        # Initialize the experiment
        print(experiment_name)
        experiment = EXPERIMENTS[experiment_name](rng=np.random.RandomState(args.seed),
                                                  tiny_clusters=tiny_clusters,
                                                  balanced_db=balanced_db)

        # Create folders and file for logging
        experiment_path = create_folder(path_to_main_file, experiment_name)
        file = open(os.path.join(experiment_path, 'out.txt'), 'w')

        # Initialize results dictionaries
        scores_dict_f1 = defaultdict(list)
        scores_test_dict_f1 = defaultdict(list)
        scores_dict_auc = defaultdict(list)
        scores_test_dict_auc = defaultdict(list)
        scores_queries_dict_f1 = defaultdict(list)
        false_mistakes_dict = defaultdict(list)

        # Initialize the learning loop
        loop = LearningLoop(experiment, args.n_clusters, args.max_iter, experiment_path,
                            file, args.plots_off, use_weights, use_labels)

        # Split the data into initially labeled (known), pool of unlabeled to choose from (train) and unlabeled for test
        folds = experiment.split(prop_known=experiment.prop_known, n_splits=args.n_folds, split_seed=args.seed)
        for k, (known_idx, train_idx, test_idx) in enumerate(folds):
            # Remove duplicates
            known_idx, train_idx, test_idx = np.unique(known_idx), np.unique(train_idx), np.unique(test_idx)

            # Plot the points
            if not args.plots_off:
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

        dump(experiment_path + '__scores.pickle', {
            'train_f1': scores_dict_f1,
            'test_f1': scores_test_dict_f1,
            'train_auc': scores_dict_auc,
            'test_auc': scores_test_dict_auc,
            'queries_f1': scores_queries_dict_f1,
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
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)

    parser.add_argument('--experiments',
                        nargs='+',
                        choices=sorted(EXPERIMENTS.keys()),
                        default=EXPERIMENTS.keys(),
                        help='The names of the experiments to be performed')
    parser.add_argument('--strategies',
                        nargs='+',
                        choices=sorted(STRATEGIES),
                        default=STRATEGIES,
                        help='The query selection strategies to be executed')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='RNG seed')
    parser.add_argument('--max_iter',
                        type=int,
                        default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--n_folds',
                        type=int,
                        default=3,
                        help="Number of cross-validation folds")
    parser.add_argument('--n_clusters',
                        type=int,
                        default=10,
                        help="Number of clusters for XGL(clustering)")
    parser.add_argument('--plots_off',
                        type=bool,
                        default=True,
                        help="Whether to plot additional graphs")
    parser.add_argument('--betas',
                        nargs='+',
                        default=[1],
                        help="The beta values for density weighted AL")
    parser.add_argument('--thetas_rules',
                        nargs='+',
                        default=[100.0],
                        help="The theta values for softmax in XGL(rules)")
    parser.add_argument('--thetas_xgl',
                        nargs='+',
                        default=[1.0],
                        help="The theta values for softmax in XGL(clustering)")

    args = parser.parse_args()
    strategies = create_strategies_list(args)

    np.random.seed(args.seed)
    run_experiment(strategies, args)


if __name__ == '__main__':
    main()