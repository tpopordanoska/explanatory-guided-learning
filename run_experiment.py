from src import *
from src.experiments import *

LEARNERS = {
    'svm_rbf': SVM(model_name='svm_rbf'),
    # 'svm': SVM(model_name='svm'),
    # 'l1svm': SVM(model_name='l1svm'),
    # 'lr': LogRegression(),
    # 'gbc': GradientBoosting()
}

path = create_folders()
# methods = ["sq_random"]
methods = ["al_least_confident", "sq_random", 'optimal_user_t1', 'optimal_user_t2', 'optimal_user_t3']
clust_use_labels = True
balanced_db = False
tiny_clusters = True
max_iter = 100
n_folds = 5
n_clusters = 20

seeds=[0]
# seeds = [0, 25, 42, 64, 100]
scores_al, scores_sq, scores_ou_t1, scores_ou_t2, scores_ou_t3 = [], [], [], [], []
test_scores_al, test_scores_sq, test_scores_ou_t1, test_scores_ou_t2, test_scores_ou_t3 = [], [], [], [], []

def get_mean_and_std(scores_arr):
    if scores_arr == []:
        return [], []
    smallest_len = min([len(x) for x in scores_arr])
    scores = [scores[:smallest_len] for scores in scores_arr]
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0) / np.sqrt(n_folds)

    return mean_scores, std_scores

EXPERIMENTS = {
    "breast-cancer": lambda rng: BreastCancer(rng),
    "habermans-survival": lambda rng: HabermansSurvival(rng),
    "adult": lambda rng: Adult(rng),
    "banknote-auth": lambda rng: BanknoteAuth(rng),
    "synthetic": lambda balanced_db, tiny_clusters, rng: Synthetic(balanced_db, tiny_clusters, rng)
}

for seed in seeds:
    rng = np.random.RandomState(seed)

    experiment = EXPERIMENTS["adult"](rng)
    plot_points(experiment.X, experiment.y, "Initial points", path)

    # Split the data into labeled and unlabeled
    folds = experiment.split(prop_known=experiment.prop_known, n_splits=n_folds)
    for k, (known_idx, train_idx, test_idx) in enumerate(folds):
        print('fold {} : #known {}, #train {}, #test {}'.format(k + 1, len(known_idx), len(train_idx), len(test_idx)))

        # for learner in LEARNERS:
        for method in methods:
            model = SVM(model_name='svm_rbf', rng=rng)
            print("Running model: ", model.model_name)
            print("Method", method)
            acc_scores, test_acc_scores = \
                ActiveLearningLoop().run(model, experiment, known_idx, train_idx, test_idx, how_many_clusters=n_clusters, max_iter=max_iter, method=method, path=path)

            if method == 'al_least_confident':
                scores_al.append(acc_scores)
                test_scores_al.append(test_acc_scores)
            elif method == 'sq_random':
                scores_sq.append(acc_scores)
                test_scores_sq.append(test_acc_scores)
            elif method == "optimal_user_t1":
                scores_ou_t1.append(acc_scores)
                test_scores_ou_t1.append(test_acc_scores)
            elif method == "optimal_user_t2":
                scores_ou_t2.append(acc_scores)
                test_scores_ou_t2.append(test_acc_scores)
            elif method == "optimal_user_t3":
                scores_ou_t3.append(acc_scores)
                test_scores_ou_t3.append(test_acc_scores)

            # points = concatenate_data(X_train_new, y_train_new, X_unlabeled_new, y_unlabeled_new, y_pred)

            # run_kmeans(points, 10, clust_use_labels)
            # run_kmedoids(points, 10, clust_use_labels, rng)

al_mean, al_std = get_mean_and_std(scores_al)
test_al_mean, test_al_std = get_mean_and_std(test_scores_al)
sq_mean, sq_std = get_mean_and_std(scores_sq)
test_sq_mean, test_sq_std = get_mean_and_std(test_scores_sq)
ou_mean_t1, ou_std_t1 = get_mean_and_std(scores_ou_t1)
test_ou_mean_t1, test_ou_std_t1 = get_mean_and_std(test_scores_ou_t1)
ou_mean_t2, ou_std_t2 = get_mean_and_std(scores_ou_t2)
test_ou_mean_t2, test_ou_std_t2 = get_mean_and_std(test_scores_ou_t2)
ou_mean_t3, ou_std_t3 = get_mean_and_std(scores_ou_t3)
test_ou_mean_t3, test_ou_std_t3 = get_mean_and_std(test_scores_ou_t3)

mean_acc_array = np.array([al_mean, sq_mean, ou_mean_t1, ou_mean_t2, ou_mean_t3])
std_acc_array = np.array([al_std, sq_std, ou_std_t1, ou_std_t2, ou_std_t3])
test_mean_acc_array = np.array([test_al_mean, test_sq_mean, test_ou_mean_t1, test_ou_mean_t2, test_ou_mean_t3])
test_std_acc_array = np.array([test_al_std, test_sq_std, test_ou_std_t1, test_ou_std_t2, test_ou_std_t3])

plot_acc(mean_acc_array, std_acc_array, labels=methods, title=experiment.name + " F1 score on train set as a function of # instances queried", path=path)
plot_acc(test_mean_acc_array, test_std_acc_array, labels=methods, title=experiment.name + "F1 score on test set as a function of # instances queried", path=path)

