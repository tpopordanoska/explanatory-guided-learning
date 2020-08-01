class Normalizer(object):

    def __init__(self, normalizer):
        self.normalizer = normalizer

    def normalize_all(self, X_known, X_train, X_test):
        X_known = self.normalizer.fit_transform(X_known)
        X_train = self.normalizer.transform(X_train)
        X_test = self.normalizer.transform(X_test)
        return X_known, X_train, X_test

    def inverse_all(self, X_known, X_train, X_test):
        X_known = self.normalizer.inverse_transform(X_known)
        X_train = self.normalizer.inverse_transform(X_train)
        X_test = self.normalizer.inverse_transform(X_test)
        return X_known, X_train, X_test

    def normalize_known_train(self, X_known, X_train):
        X_known = self.normalizer.fit_transform(X_known)
        X_train = self.normalizer.transform(X_train)
        return X_known, X_train

    def normalize(self, X):
        return self.normalizer.fit_transform(X)

    def inverse(self, X):
        return self.normalizer.inverse_transform(X)
