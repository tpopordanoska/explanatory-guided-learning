from src import *
from src.experiments import *
from sklearn.model_selection import cross_val_score

EXPERIMENTS = {
    "habermans-survival": lambda rng: HabermansSurvival(rng),
    "breast-cancer": lambda rng: BreastCancer(rng),
    "banknote-auth": lambda rng: BanknoteAuth(rng),
    "synthetic": lambda balanced_db, tiny_clusters, rng: Synthetic(balanced_db, tiny_clusters, rng),
    "adult": lambda rng: Adult(rng)
}

def get_mean_and_std(scores_dict):
    if not scores_dict:
        return {}, {}
    scores_dict_mean = {}
    scores_dict_std = {}
    for method, scores in scores_dict.items():
        if scores:
            smallest_len = min([len(x) for x in scores])
            scores_smallest_len = [s[:smallest_len] for s in scores]
            scores_dict_mean[method] = np.mean(scores_smallest_len, axis=0)
            scores_dict_std[method] = np.std(scores_smallest_len, axis=0) / np.sqrt(n_folds)

    return scores_dict_mean, scores_dict_std


def get_passive_f1(experiment, file):
    exp_model = experiment.model._model
    model = SVC(gamma=exp_model.gamma, C=exp_model.C, random_state=exp_model.random_state)
    scores = cross_val_score(model, experiment.X, experiment.y, cv=5, scoring="f1")
    file.write("Passive accuracy for {}: {:.2f} (+/- {:.2f}) ".format(experiment.name, scores.mean(), scores.std() * 2))
    return scores


path = create_folders()
file = open(path + '\\out.txt', 'w')
# methods = ["sq_random"]
methods = ["al_least_confident", "sq_random", 'optimal_user_t1', 'optimal_user_t2', 'optimal_user_t3']
clust_use_labels = True
balanced_db = True
tiny_clusters = True
max_iter = 100
n_folds = 5
n_clusters = 40
plots_off = True

seeds=[0]
# seeds = [0, 25, 42, 64, 100]
scores_dict = {}
scores_test_dict = {}

for seed in seeds:
    rng = np.random.RandomState(seed)

    experiment = EXPERIMENTS["habermans-survival"](rng)
    # experiment = EXPERIMENTS["breast-cancer"](rng)
    # experiment = EXPERIMENTS["banknote-auth"](rng)
    # experiment = EXPERIMENTS["synthetic"](balanced_db, tiny_clusters, rng)
    # experiment = EXPERIMENTS["adult"](rng)
    # plot_points(experiment.X, experiment.y, "Initial points", path)

    learning_loop = ActiveLearningLoop(experiment, n_clusters, max_iter, path, file, plots_off)

    # Split the data into labeled and unlabeled
    folds = experiment.split(prop_known=experiment.prop_known, n_splits=n_folds)
    for k, (known_idx, train_idx, test_idx) in enumerate(folds):
        # Remove duplicates
        known_idx, train_idx, test_idx = np.unique(known_idx), np.unique(train_idx), np.unique(test_idx)
        file.write('fold {} : #known {}, #train {}, #test {} \n'
                   .format(k + 1, len(known_idx), len(train_idx), len(test_idx)))
        _, counts_known = np.unique(experiment.y[known_idx], return_counts=True)
        _, counts_train = np.unique(experiment.y[train_idx], return_counts=True)
        file.write("Known bincount: {}\n".format(counts_known))
        file.write("Train bincount: {}\n".format(counts_train))

        for method in methods:
            file.write("Running model: {} Method: {} \n".format(experiment.model.name, method))
            acc_scores, test_acc_scores = learning_loop.run(method, known_idx, train_idx, test_idx)

            if method not in scores_dict:
                scores_dict[method] = [acc_scores]
                scores_test_dict[method] = [test_acc_scores]
            else:
                scores_dict[method].append(acc_scores)
                scores_test_dict[method].append(test_acc_scores)

scores_dict_mean, scores_dict_std = get_mean_and_std(scores_dict)
scores_test_dict_mean, scores_test_dict_std = get_mean_and_std(scores_test_dict)

f1_score_passive = get_passive_f1(experiment, file)
file.write("Passive accuracy for {}:  {:.2f} (+/- {:.2f}) "
           .format(experiment.name, f1_score_passive.mean(), f1_score_passive.std() * 2))

plot_acc(scores_dict_mean, scores_dict_std, f1_score_passive, title= "{} {} F1 score on train set using {}"
         .format(experiment.model.name, experiment.name, str(n_clusters)), path=path)

plot_acc(scores_test_dict_mean, scores_test_dict_std, f1_score_passive, title="{} {} F1 score on test set using {}"
         .format(experiment.model.name, experiment.name, str(n_clusters)), path=path)
