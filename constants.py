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

LABELS_LOOKUP = {
    "random": "Random sampling",
    "al_least_confident": "AL(least confident)",
    "al_density_weighted_2": "AL(density weighted) (beta=2)",
    "al_density_weighted_1": "AL(density weighted) (beta=1)",
    "al_density_weighted_0.5": "AL(density weighted) (beta=0.5)",
    "sq_random": "Guided Learning",
    "xgl_1000.0": "XGL(clustering) (theta=1000)",
    "xgl_100.0": "XGL(clustering) (theta=100)",
    "xgl_5": "XGL(clustering) (theta=5)",
    "xgl_10.0": "XGL(clustering) (theta=10)",
    "xgl_1.0": "XGL(clustering) (theta=1)",
    "xgl_0.1": "XGL(clustering) (theta=0.1)",
    "xgl_0.01": "XGL(clustering) (theta=0.01)",
    "rules_hierarchy_1000.0": "XGL(rules_hierarchy) (theta=1000)",
    "rules_hierarchy_100.0": "XGL(rules_hierarchy) (theta=100)",
    "rules_hierarchy_10.0": "XGL(rules_hierarchy) (theta=10)",
    "rules_hierarchy_1.0": "XGL(rules_hierarchy) (theta=1)",
    "rules_hierarchy_0.1": "XGL(rules_hierarchy) (theta=0.1)",
    "rules_hierarchy_0.01": "XGL(rules_hierarchy) (theta=0.01)",
    "rules_1000.0": "XGL(rules) (theta=1000)",
    "rules_100.0": "XGL(rules) (theta=100)",
    "rules_10.0": "XGL(rules) (theta=10)",
    "rules_1.0": "XGL(rules) (theta=1)",
    "rules_0.1": "XGL(rules) (theta=0.1)",
    "rules_0.01": "XGL(rules) (theta=0.01)",
    "rules_hierarchy": "XGL(rules_hierarchy)",
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
