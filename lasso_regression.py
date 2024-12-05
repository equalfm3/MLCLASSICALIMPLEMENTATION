import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Add intercept term
        X = np.column_stack([np.ones(n_samples), X])
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features + 1)
        
        # Coordinate descent
        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features + 1):
                if j == 0:
                    self.coef_[j] = np.mean(y - np.dot(X[:, 1:], self.coef_[1:]))
                else:
                    X_j = X[:, j]
                    y_pred = np.dot(X, self.coef_) - self.coef_[j] * X_j
                    r_j = y - y_pred
                    arg1 = np.dot(X_j, r_j)
                    arg2 = self.alpha * n_samples
                    self.coef_[j] = self._soft_threshold(arg1, arg2) / (X_j**2).sum()

            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
        
        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v