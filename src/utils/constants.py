from src.experiments import *
from src.sampling_strategies import *

# List of sampling strategies
STRATEGIES = ["random",
              "al_lc",
              "al_dw",
              "sq_random",
              "xgl_clusters",
              'xgl_rules',
              'xgl_rules_simple_tree',
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
    "heart": lambda **kwargs: Heart(**kwargs),
    "wine": lambda **kwargs: Wine(**kwargs),
    "covertype": lambda **kwargs: Covertype(**kwargs),
    "kdd_cup99": lambda **kwargs: KddCup99(**kwargs),
    "blood_transfusion": lambda **kwargs: BloodTransfusion(**kwargs),
    "steel_plate_faults": lambda **kwargs: SteelPlateFaults(**kwargs),
    "phoneme": lambda **kwargs: Phoneme(**kwargs),
    "iris": lambda **kwargs: Iris(**kwargs),
    "glass": lambda **kwargs: Glass(**kwargs)
    "heloc": Heloc,
    "newsgroups-religion": NewsgroupsReligion,
    "newsgroups-science": NewsgroupsScience,
    "magic": Magic,
    "risk": Risk,
}

METHODS = {
    "random": lambda **kwargs: RandomSampling(**kwargs),
    "al_lc": lambda **kwargs: LeastConfidentAL(**kwargs),
    "al_dw": lambda **kwargs: DensityWeightedAL(**kwargs),
    "sq_random": lambda **kwargs: GuidedLearning(**kwargs),
    "xgl_clusters": lambda **kwargs: XglClusters(**kwargs),
    "xgl_rules": lambda **kwargs: XglRules(**kwargs),
    "xgl_rules_hierarchy": lambda **kwargs: XglRules(hierarchy=True, **kwargs),
    "xgl_rules_simple_tree": lambda **kwargs: XglRules(simple_tree=True, **kwargs)
}

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:brown']

MARKERS = ['o', 'v', 's', 'D', 'p', '*', 'd', '<', '<', "x"]

COLORS_RED = ['#840000', 'red', '#fc824a']

LINESTYLE = ['solid', 'dotted', 'dashed']
