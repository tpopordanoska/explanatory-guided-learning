
from src import *
from src.experiment_synthethic import *

LEARNERS = {
    'svm_rbf': SVM(model_name='svm_rbf'),
    # 'svm': SVM(model_name='svm'),
    # 'l1svm': SVM(model_name='l1svm'),
    # 'lr': LogRegression(),
    # 'gbc': GradientBoosting()
}

path = create_folders()
# methods = ["sq_random"]
methods = ["al_least_confident", "sq_random", 'optimal_user']
clust_use_labels = True
balanced_db = False
tiny_clusters = True
max_iter = 50
n_folds = 10

seeds=[0]
# seeds = [0, 25, 42, 64, 100]
scores_al = []
scores_sq = []
scores_ou = []

test_scores_al = []
test_scores_sq = []
test_scores_ou = []

def get_mean_and_std(scores_arr):
    smallest_len = min([len(x) for x in scores_arr])
    scores = [scores[:smallest_len] for scores in scores_arr]
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0) / np.sqrt(n_folds)

    return mean_scores, std_scores

for seed in seeds:
    rng = np.random.RandomState(seed)

    synthetic_exp = Synthetic(balanced_db, tiny_clusters, rng)
    plot_points(synthetic_exp.X, synthetic_exp.y, "Initial points", path)

    # Split the data into labeled and unlabeled
    folds = synthetic_exp.split(prop_known=0.2, n_splits=n_folds)
    for k, (known_idx, train_idx, test_idx) in enumerate(folds):
        print('fold {} : #known {}, #train {}, #test {}'.format(k + 1, len(known_idx), len(train_idx), len(test_idx)))

        # for learner in LEARNERS:
        for method in methods:
            model = SVM(model_name='svm_rbf', rng=rng)
            print("Running model: ", model.model_name)
            print("Method", method)
            acc_scores, test_acc_scores = \
                ActiveLearningLoop().run(model, synthetic_exp, known_idx, train_idx, test_idx, max_iter=max_iter, method=method, path=path)

            if method == 'al_least_confident':
                scores_al.append(acc_scores)
                test_scores_al.append(test_acc_scores)
            elif method == 'sq_random':
                scores_sq.append(acc_scores)
                test_scores_sq.append(test_acc_scores)
            elif method == "optimal_user":
                scores_ou.append(acc_scores)
                test_scores_ou.append(test_acc_scores)

            # points = concatenate_data(X_train_new, y_train_new, X_unlabeled_new, y_unlabeled_new, y_pred)

            # run_kmeans(points, 10, clust_use_labels)
            # run_kmedoids(points, 10, clust_use_labels, rng)

al_mean, al_std = get_mean_and_std(scores_al)
test_al_mean, test_al_std = get_mean_and_std(test_scores_al)
sq_mean, sq_std = get_mean_and_std(scores_sq)
test_sq_mean, test_sq_std = get_mean_and_std(test_scores_sq)
ou_mean, ou_std = get_mean_and_std(scores_ou)
test_ou_mean, test_ou_std = get_mean_and_std(test_scores_ou)

mean_acc_array = np.array([al_mean, sq_mean, ou_mean])
std_acc_array = np.array([al_std, sq_std, ou_std])
test_mean_acc_array = np.array([test_al_mean, test_sq_mean, test_ou_mean])
test_std_acc_array = np.array([test_al_std, test_sq_std, test_ou_std])

plot_acc(mean_acc_array, std_acc_array, labels=methods, title="F1 score on train set as a function of # instances queried", path=path)
plot_acc(test_mean_acc_array, test_std_acc_array, labels=methods, title="F1 score on test set as a function of # instances queried", path=path)

