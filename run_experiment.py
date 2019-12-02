
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
cross_val = 5

seeds=[0]
# seeds = [0, 25, 42, 64, 100]
scores_al = []
scores_sq = []
scores_ou = []

test_scores_al = []
test_scores_sq = []
test_scores_ou = []

for seed in seeds:
    rng = np.random.RandomState(seed)

    synthetic_exp = Synthetic(balanced_db, tiny_clusters, rng)
    plot_points(synthetic_exp.X, synthetic_exp.y, "Initial points", path)

    # Split the data into labeled and unlabeled
    folds = synthetic_exp.split(prop_known=0.2, n_splits=cross_val)
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

smallest_len_al = min([len(x) for x in scores_al])
smallest_len_sq = min([len(x) for x in scores_sq])
smallest_len_ou = min([len(x) for x in scores_ou])

scores_al = [scores[:smallest_len_al] for scores in scores_al]
scores_sq = [scores[:smallest_len_sq] for scores in scores_sq]
scores_ou = [scores[:smallest_len_ou] for scores in scores_ou]

mean_scores_al = np.mean(scores_al, axis=0)
mean_scores_sq = np.mean(scores_sq, axis=0)
mean_scores_ou = np.mean(scores_ou, axis=0)

test_smallest_len_al = min([len(x) for x in test_scores_al])
test_smallest_len_sq = min([len(x) for x in test_scores_sq])
test_smallest_len_ou = min([len(x) for x in test_scores_ou])

test_scores_al = [scores[:test_smallest_len_al] for scores in test_scores_al]
test_scores_sq = [scores[:test_smallest_len_sq] for scores in test_scores_sq]
test_scores_ou = [scores[:test_smallest_len_ou] for scores in test_scores_ou]

test_mean_scores_al = np.mean(test_scores_al, axis=0)
test_mean_scores_sq = np.mean(test_scores_sq, axis=0)
test_mean_scores_ou = np.mean(test_scores_ou, axis=0)


acc_array = np.array([mean_scores_al, mean_scores_sq, mean_scores_ou])
test_acc_array = np.array([test_mean_scores_al, test_mean_scores_sq, test_mean_scores_ou])

plot_acc(acc_array, methods, title="F1 score on train set as a function of # instances queried", path=path)
plot_acc(test_acc_array, methods, title="F1 score on test set as a function of # instances queried", path=path)