import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture:
    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights_ = np.full(self.n_components, 1/self.n_components)
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = [np.cov(X.T) for _ in range(self.n_components)]
        
        # Initialize log-likelihood
        self.log_likelihood_ = -np.inf
        
        for _ in range(self.max_iter):
            # E-step
            resp = self._e_step(X)
            
            # M-step
            self._m_step(X, resp)
            
            # Compute log-likelihood
            new_log_likelihood = self._compute_log_likelihood(X)
            
            # Check for convergence
            if np.abs(new_log_likelihood - self.log_likelihood_) < self.tol:
                break
            
            self.log_likelihood_ = new_log_likelihood
        
        return self

    def _e_step(self, X):
        resp = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * multivariate_normal.pdf(X, self.means_[k], self.covariances_[k])
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X, resp):
        N = resp.sum(axis=0)
        self.weights_ = N / X.shape[0]
        self.means_ = np.dot(resp.T, X) / N[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(resp[:, k] * diff.T, diff) / N[k]

    def _compute_log_likelihood(self, X):
        return np.sum(np.log(np.sum([w * multivariate_normal.pdf(X, m, c) 
                                     for w, m, c in zip(self.weights_, self.means_, self.covariances_)], axis=0)))

    def predict(self, X):
        resp = self._e_step(X)
        return np.argmax(resp, axis=1)