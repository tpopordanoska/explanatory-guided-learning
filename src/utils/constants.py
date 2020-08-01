from src.experiments import *
from src.sampling_strategies import *

# List of sampling strategies
STRATEGIES = ["random",
              "al_lc",
              "al_dw",
              "sq_random",
              "xgl_clusters",
              'xgl_rules',
              'xgl_rules_hierarchy']

EXPERIMENTS = {
    "german": lambda **kwargs: German(**kwargs),
    "breast-cancer": lambda **kwargs: BreastCancer(**kwargs),
    "banknote-auth": lambda **kwargs: BanknoteAuth(**kwargs),
    "synthetic": lambda **kwargs: Synthetic(**kwargs),
    "adult": lambda **kwargs: Adult(**kwargs),
    "credit": lambda **kwargs: Credit(**kwargs),
    "australian": lambda **kwargs: Australian(**kwargs),
    "hepatitis": lambda **kwargs: Hepatitis(**kwargs),
    "heart": lambda **kwargs: Heart(**kwargs)
}

METHODS = {
    "random": lambda **kwargs: RandomSampling(**kwargs),
    "al_lc": lambda **kwargs: LeastConfidentAL(**kwargs),
    "al_dw": lambda **kwargs: DensityWeightedAL(**kwargs),
    "sq_random": lambda **kwargs: GuidedLearning(**kwargs),
    "xgl_clusters": lambda **kwargs: XglClusters(**kwargs),
    "xgl_rules": lambda **kwargs: XglRules(**kwargs),
    "xgl_rules_hierarchy": lambda **kwargs: XglRules(hierarchy=True, **kwargs),
}

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:brown']

MARKERS = ['o', 'v', 's', 'D', 'p', '*', 'd', '<', '<', "x"]

COLORS_RED = ['#840000', 'red', '#fc824a']

LINESTYLE = ['solid', 'dotted', 'dashed']
