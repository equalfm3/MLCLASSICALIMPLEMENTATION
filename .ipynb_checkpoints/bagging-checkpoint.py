import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class BaggingClassifier:
    def __init__(self, n_estimators=10, max_samples=1.0, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.estimators_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=int(n_samples * self.max_samples), replace=True)
            else:
                indices = np.random.choice(n_samples, size=int(n_samples * self.max_samples), replace=False)
            
            estimator = DecisionTreeClassifier()
            estimator.fit(X[indices], y[indices])
            self.estimators_.append(estimator)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

class BaggingRegressor:
    def __init__(self, n_estimators=10, max_samples=1.0, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.estimators_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=int(n_samples * self.max_samples), replace=True)
            else:
                indices = np.random.choice(n_samples, size=int(n_samples * self.max_samples), replace=False)
            
            estimator = DecisionTreeRegressor()
            estimator.fit(X[indices], y[indices])
            self.estimators_.append(estimator)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        return np.mean(predictions, axis=0)