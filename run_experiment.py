from sklearn.datasets.samples_generator import make_blobs

from src import *

LEARNERS = {
    'svm_rbf': SVM(model_name='svm_rbf'),
    # 'svm': SVM(model_name='svm'),
    # 'l1svm': SVM(model_name='l1svm'),
    # 'lr': LogRegression(),
    # 'gbc': GradientBoosting()
}

methods = ["al_least_confident", "sq_random"]
clust_use_labels = True
seeds = [0, 25, 42, 64, 100]
scores_al = []
scores_sq = []

for seed in seeds:
    rng = np.random.RandomState(seed)
    # Generate mock data with balanced number of positive and negative examples
    # X, y = generate_points(5, 40, 1)

    # Generate mock data with rare grid class
    X_pos, y_pos = generate_positive(5)
    X_neg, y_neg = generate_negative(5, 40, rng)

    known_idx, unknown_idx = split_data(X_pos, y_pos, 0.6, rng)

    # Generate tiny clusters (0 to 5 points) around the positive points
    centers = X_pos[known_idx]
    n_samples = rng.randint(0, 5, size=len(centers))
    cluster_std = rng.uniform(0, 0.1, size=len(centers))

    Xpos, _ = make_blobs(n_samples=n_samples, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
    ypos = np.ones((len(Xpos)), dtype=int)

    X = np.concatenate((Xpos, X_neg), axis=0)
    y = np.concatenate((ypos, y_neg), axis=0)

    # X = np.concatenate((X_pos[known_idx], X_neg), axis=0)
    # y = np.concatenate((y_pos[known_idx], y_neg), axis=0)
    plot_points(X, y, "Initial points")

    # Split the data into labeled and unlabeled
    labeled_indicies, unlabeled_indicies = split_data(X, y, 0.8, rng)
    X_labeled, y_labeled = X[labeled_indicies], y[labeled_indicies]
    X_unlabeled, y_unlabeled = X[unlabeled_indicies], y[unlabeled_indicies]

    print("Unlabeled points: ", X_unlabeled.shape, " y: ", y_unlabeled.shape)
    print("Labeled points: ", X_labeled.shape, " y: ", y_labeled.shape)
    plot_points(X_labeled, y_labeled, "The labeled points")


    # for learner in LEARNERS:
    for method in methods:
        model = SVM(model_name='svm_rbf', rng=rng)
        print("Running model: ", model.model_name)
        print("Method", method)
        X_train_new, y_train_new, X_unlabeled_new, y_unlabeled_new, y_pred, acc_scores = \
            ActiveLearningLoop().run(model, X_labeled,
                                     y_labeled, X_unlabeled, y_unlabeled, max_iter=12, method=method)
        if method == 'al_least_confident':
            scores_al.append(acc_scores)
        elif method == 'sq_random':
            scores_sq.append(acc_scores)

        # Plot the predictions
        # plot_decision_surface(model,
        #                       X_train_new,
        #                       y_train_new,
        #                       X_unlabeled_new,
        #                       y_unlabeled_new,
        #                       y_pred=y_pred,
        #                       title="Predictions of the model " + model.model_name)

        points = concatenate_data(X_train_new, y_train_new, X_unlabeled_new, y_pred)

        # run_kmeans(points, 10, clust_use_labels)
        run_kmedoids(points, 10, clust_use_labels, rng)

plot_acc(np.mean(scores_al, axis=0), np.mean(scores_sq, axis=0), "Active learning", "Search queries")
