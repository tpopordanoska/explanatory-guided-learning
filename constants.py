from src.experiments import *

# List of sampling strategies
STRATEGIES = ["random",
              "al_least_confident",
              "al_density_weighted",
              "sq_random",
              "xgl",
              'rules',
              'rules_hierarchy']

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

LABELS_LOOKUP = {
    "random": "Random",
    "al_least_confident": "AL (unc.)",
    "al_density_weighted_2": "AL (repr.)",
    "al_density_weighted_1": "AL (repr.)",
    "al_density_weighted_0.5": "AL (repr.)",
    "sq_random": "GL",
    "xgl_1000.0": "XGL(clustering) (θ=1000)",
    "xgl_100.0": "XGL(clustering) (θ=100)",
    "xgl_5": "XGL(clustering) (θ=5)",
    "xgl_10.0": "XGL(clustering) (θ=10)",
    "xgl_1.0": "XGL(clustering) (θ=1)",
    "xgl_0.1": "XGL(clustering) (θ=0.1)",
    "xgl_0.01": "XGL(clustering) (θ=0.01)",
    "rules_hierarchy_1000.0": "XGL (hier.) (θ=1000)",
    "rules_hierarchy_100.0": "XGL(hier.) (θ=100)",
    "rules_hierarchy_10.0": "XGL(hier.) (θ=10)",
    "rules_hierarchy_1.0": "XGL(hier.) (θ=1)",
    "rules_hierarchy_0.1": "XGL(hier.) (θ=0.1)",
    "rules_hierarchy_0.01": "XGL(hier.) (θ=0.01)",
    "rules_1000.0": "XGL (θ=1000)",
    "rules_100.0": "XGL (θ=100)",
    "rules_10.0": "XGL (θ=10)",
    "rules_1.0": "XGL (θ=1)",
    "rules_0.1": "XGL (θ=0.1)",
    "rules_0.01": "XGL (θ=0.01)",
    "rules_hierarchy": "XGL (hier.)",
    "rules": "XGL (rules)",
    "10": "10 prototypes",
    "30": "30 prototypes",
    "50": "50 prototypes",
    "100": "100 prototypes"
}


COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:brown']

MARKERS = ['o', 'v', 's', 'D', 'p', '*', 'd', '<', '<', "x"]

COLORS_RED = ['#840000', 'red', '#fc824a']

LINESTYLE = ['solid', 'dotted', 'dashed']
