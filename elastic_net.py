import numpy as np

class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def _soft_thresholding(self, x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Add intercept term
        X = np.c_[np.ones((n_samples, 1)), X]
        
        self.coef_ = np.zeros(n_features + 1)
        
        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features + 1):
                if j == 0:
                    self.coef_[j] = np.mean(y - np.dot(X[:, 1:], self.coef_[1:]))
                else:
                    z = np.dot(X[:, j], y - np.dot(X, self.coef_) + self.coef_[j] * X[:, j])
                    z = np.clip(z, -1e15, 1e15)  # Prevent overflow
                    lambda1 = self.alpha * self.l1_ratio
                    lambda2 = self.alpha * (1 - self.l1_ratio)
                    
                    if j == 0:  # intercept
                        self.coef_[j] = z / (X[:, j] ** 2).sum()
                    else:
                        denominator = (X[:, j] ** 2).sum() + lambda2
                        if denominator != 0:
                            self.coef_[j] = self._soft_thresholding(z, lambda1) / denominator
                        else:
                            self.coef_[j] = 0
            
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        return self.intercept_ + np.dot(X, self.coef_)