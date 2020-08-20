import time
import warnings

import numpy as np

from src.learning_loop import LearningLoop
from src.results.all_results import Results
from src.utils import create_folders, initialize_experiment, write_to_file, create_strategies_list
from src.utils.arguments import get_main_args
from src.utils.plotting import plot_initial_points
from src.utils.clustering import introduce_uu
from src.experiments import Synthetic

warnings.filterwarnings('ignore')


def run_experiments(strategies, args):
    results_path = create_folders()
    for experiment_name in args.experiments:
        experiment = initialize_experiment(experiment_name, args.seed, results_path)
        if not isinstance(experiment, Synthetic):
            introduce_uu(experiment)
        loop = LearningLoop()
        results = Results()

        folds = experiment.split(prop_known=experiment.prop_known, n_splits=args.n_folds, split_seed=args.seed)
        for k, (known_idx, train_idx, test_idx) in enumerate(folds):
            # Remove duplicates
            known_idx, train_idx, test_idx = np.unique(known_idx), np.unique(train_idx), np.unique(test_idx)
            loop.update_data_split(known_idx, train_idx, test_idx)

            if args.plots_on:
                plot_initial_points(known_idx, test_idx, experiment)

            # Run the experiment for every strategy
            for strategy in strategies:
                print(strategy)
                start = time.time()
                running_instance = loop.run(args, experiment, strategy)
                end = time.time()

                write_to_file(loop, args, k, experiment, strategy, str(end - start))
                results.collect(strategy, running_instance)
                results.export(experiment.path)


def main():
    args = get_main_args()
    strategies = create_strategies_list(args)

    np.random.seed(args.seed)
    run_experiments(strategies, args)


if __name__ == '__main__':
    main()
