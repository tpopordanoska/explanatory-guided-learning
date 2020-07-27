import argparse
from constants import EXPERIMENTS, STRATEGIES


def get_main_args():
    parser = create_parser()
    add_strategies_options_args(parser)
    add_num_iter_folds_clusters_args(parser)
    add_experiments_and_seed_args(parser)
    add_plotting_option_arg(parser)

    return parser.parse_args()


def get_drawing_args(folder):
    parser = create_parser()
    add_strategies_options_args(parser)
    add_folder_arg(parser, folder)

    return parser.parse_args()


def get_experiment_args():
    parser = create_parser()
    add_experiments_and_seed_args(parser)

    return parser.parse_args()


def create_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def add_strategies_options_args(parser):
    parser.add_argument('--strategies',
                        nargs='+',
                        choices=sorted(STRATEGIES),
                        default=STRATEGIES)
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

    return parser


def add_num_iter_folds_clusters_args(parser):
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


def add_experiments_and_seed_args(parser):
    parser.add_argument('--experiments',
                        nargs='+',
                        choices=sorted(EXPERIMENTS.keys()),
                        default=EXPERIMENTS.keys(),
                        help='The names of the experiments to be performed')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='RNG seed')


def add_folder_arg(parser, folder):
    parser.add_argument('--folder',
                        type=str,
                        default=folder[-1],
                        help="The name of the folder where the .pickle file is stored")


def add_plotting_option_arg(parser):
    parser.add_argument('--plots_on',
                        type=bool,
                        default=False,
                        help="Whether to plot additional graphs")
