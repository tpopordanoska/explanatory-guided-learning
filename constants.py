from src.experiments import *

# List of sampling strategies
STRATEGIES = ["random",
              "al_lc",
              "al_dw",
              "sq_random",
              "xgl_clusters",
              'xgl_rules',
              'xgl_rules_hierarchy',
              "optimal"]

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
    "al_lc": "AL (unc.)",
    "al_dw_2": "AL (repr.)",
    "al_dw_1": "AL (repr.)",
    "al_dw_0.5": "AL (repr.)",
    "sq_random": "GL",
    "xgl_clusters_1000.0": "XGL(clustering) (θ=1000)",
    "xgl_clusters_100.0": "XGL(clustering) (θ=100)",
    "xgl_clusters_5": "XGL(clustering) (θ=5)",
    "xgl_clusters_10.0": "XGL(clustering) (θ=10)",
    "xgl_clusters_1.0": "XGL(clustering) (θ=1)",
    "xgl_clusters_0.1": "XGL(clustering) (θ=0.1)",
    "xgl_clusters_0.01": "XGL(clustering) (θ=0.01)",
    "xgl_rules_hierarchy_1000.0": "XGL (hier.) (θ=1000)",
    "xgl_rules_hierarchy_100.0": "XGL(hier.) (θ=100)",
    "xgl_rules_hierarchy_10.0": "XGL(hier.) (θ=10)",
    "xgl_rules_hierarchy_1.0": "XGL(hier.) (θ=1)",
    "xgl_rules_hierarchy_0.1": "XGL(hier.) (θ=0.1)",
    "xgl_rules_hierarchy_0.01": "XGL(hier.) (θ=0.01)",
    "xgl_rules_1000.0": "XGL (θ=1000)",
    "xgl_rules_100.0": "XGL (θ=100)",
    "xgl_rules_10.0": "XGL (θ=10)",
    "xgl_rules_1.0": "XGL (θ=1)",
    "xgl_rules_0.1": "XGL (θ=0.1)",
    "xgl_rules_0.01": "XGL (θ=0.01)",
    "xgl_rules_hierarchy": "XGL (hier.)",
    "xgl_rules": "XGL (rules)",
    "10": "10 prototypes",
    "30": "30 prototypes",
    "50": "50 prototypes",
    "100": "100 prototypes"
}


COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:brown']

MARKERS = ['o', 'v', 's', 'D', 'p', '*', 'd', '<', '<', "x"]

COLORS_RED = ['#840000', 'red', '#fc824a']

LINESTYLE = ['solid', 'dotted', 'dashed']
