import os
from src.utils import load, dump


def merge_pickles(path_res):
    results = {
        'synthetic': {},
        'adult': {},
        'australian': {},
        'banknote-auth': {},
        'breast-cancer': {},
        'credit': {},
        'german': {},
        'heart': {},
        'hepatitis': {},
    }

    folders = os.listdir(path_res)
    for folder in folders:
        path_folder = os.path.join(path_res, folder)
        pickle = [f for f in os.listdir(path_folder) if f.endswith(".pickle")][0]
        experiment = pickle.split("__")[0]
        pickle_results = load(os.path.join(path_folder, pickle))
        if results.get(experiment) == {}:
            results[experiment] = pickle_results
        else:
            experiment_dict = results.get(experiment)
            for k in experiment_dict.keys():
                experiment_dict[k].update(pickle_results[k])

    for exp in results.keys():
        exp_folder = os.path.join(path_res, exp)
        print(exp_folder + '__scores.pickle')
        dump(exp_folder + '__scores.pickle', results[exp])


merge_pickles(os.path.join(os.getcwd(), "results", "separate_rulebudget_10f"))
