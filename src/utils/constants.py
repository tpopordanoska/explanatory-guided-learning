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
    "adult": lambda **kwargs: Adult(**kwargs),
    "australian": lambda **kwargs: Australian(**kwargs),
    "banknote-auth": lambda **kwargs: BanknoteAuth(**kwargs),
    "blood_transfusion": lambda **kwargs: BloodTransfusion(**kwargs),
    "breast-cancer": lambda **kwargs: BreastCancer(**kwargs),
    "credit": lambda **kwargs: Credit(**kwargs),
    "german": lambda **kwargs: German(**kwargs),
    "glass": lambda **kwargs: Glass(**kwargs),
    "heart": lambda **kwargs: Heart(**kwargs),
    "heloc": Heloc,
    "hepatitis": lambda **kwargs: Hepatitis(**kwargs),
    "iris": lambda **kwargs: Iris(**kwargs),
    "magic": Magic,
    "newsgroups-religion": NewsgroupsReligion,
    "newsgroups-science": NewsgroupsScience,
    "phoneme": lambda **kwargs: Phoneme(**kwargs),
    "risk": Risk,
    "steel_plate_faults": lambda **kwargs: SteelPlateFaults(**kwargs),
    "synthetic": lambda **kwargs: Synthetic(**kwargs),
    "wine": lambda **kwargs: Wine(**kwargs),

    #"covertype": lambda **kwargs: Covertype(**kwargs),
    #"kdd_cup99": lambda **kwargs: KddCup99(**kwargs),
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
