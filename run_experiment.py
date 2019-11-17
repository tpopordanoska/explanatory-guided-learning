
from src import *

setting = "SQ"
method = "sq_random"
clust_use_labels = True

LEARNERS = {
    'svm_rbf': SVM(model_name='svm_rbf'),
    'svm': SVM(model_name='svm'),
    'l1svm': SVM(model_name='l1svm'),
    'lr': LogRegression(),
    'gbc': GradientBoosting()
}

# Generate mock data with balanced number of positive and negative examples
# X, y = generate_points(5, 40, 1)

# Generate mock data with rare grid class
X_pos, y_pos = generate_positive(5)
X_neg, y_neg = generate_negative(5, 40)

known_idx, unknown_idx = split_data(X_pos, y_pos, 0.6)

X = np.concatenate((X_pos[known_idx], X_neg), axis=0)
y = np.concatenate((y_pos[known_idx], y_neg), axis=0)
plot_points(X,y, "Initial points")

# Split the data into labeled and unlabeled
labeled_indicies, unlabeled_indicies = split_data(X, y, 0.8)
X_labeled, y_labeled = X[labeled_indicies], y[labeled_indicies]
X_unlabeled, y_unlabeled = X[unlabeled_indicies], y[unlabeled_indicies]

print("Unlabeled points: ", X_unlabeled.shape, " y: ", y_unlabeled.shape)
print("Labeled points: ", X_labeled.shape, " y: ", y_labeled.shape)
plot_points(X_labeled, y_labeled, "The labeled points")

for learner in LEARNERS:
    print("Running model: ", learner)
    print("Setting: ", setting)
    model = LEARNERS[learner]
    X_train_new, y_train_new, X_unlabeled_new, y_unlabeled_new, y_pred, acc_scores = \
        ActiveLearningLoop().run(model, X_labeled,
                                 y_labeled, X_unlabeled, y_unlabeled, max_iter=5, method=method)

    plot_acc(acc_scores)
    # Plot the predictions
    plot_decision_surface(model,
                          X_train_new,
                          y_train_new,
                          X_unlabeled_new,
                          y_unlabeled_new,
                          y_pred=y_pred,
                          title="Setting: " + setting + " Predictions of the model " + model.model_name)

    points = concatenate_data(X_train_new, y_train_new, X_unlabeled_new, y_pred)

    run_kmeans(points, 10, clust_use_labels)
    run_kmedoids(points, 10, clust_use_labels)
