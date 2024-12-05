import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Initialize prediction with mean of target
        self.initial_prediction = np.mean(y)
        
        # Fit weak learners
        for _ in range(self.n_estimators):
            residuals = y - self.predict(X)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Initialize prediction with log-odds of positive class
        self.initial_prediction = np.log(np.mean(y) / (1 - np.mean(y)))
        
        # Fit weak learners
        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(self.predict(X))
            residuals = y - probabilities
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

    def predict(self, X):
        log_odds = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            log_odds += self.learning_rate * tree.predict(X)
        return log_odds

    def predict_proba(self, X):
        return self._sigmoid(self.predict(X))

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))