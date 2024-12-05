import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        n_samples, n_features = X.shape
        identity = np.eye(n_features)
        identity[0, 0] = 0  # Don't regularize the intercept
        
        # Ridge regression formula: (X^T X + alpha * I)^(-1) X^T y
        self.coef_ = np.linalg.inv(X.T.dot(X) + self.alpha * identity).dot(X.T).dot(y)
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0
        
        return self

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v