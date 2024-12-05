import numpy as np
from scipy.optimize import minimize

class GaussianProcessRegressor:
    def __init__(self, kernel='rbf', alpha=1e-10, optimizer='L-BFGS-B', n_restarts_optimizer=0):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer

    def rbf_kernel(self, X1, X2, l=1.0, sigma_f=1.0):
        """Radial basis function kernel."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        def negative_log_likelihood_loss(theta):
            l, sigma_f = theta
            K = self.rbf_kernel(self.X_train, self.X_train, l, sigma_f) + \
                self.alpha * np.eye(len(self.X_train))
            return 0.5 * self.y_train.T.dot(np.linalg.inv(K)).dot(self.y_train) + \
                   0.5 * np.log(np.linalg.det(K)) + 0.5 * len(self.X_train) * np.log(2*np.pi)

        # Optimize kernel parameters
        res = minimize(negative_log_likelihood_loss, [1, 1], 
                       bounds=((1e-5, None), (1e-5, None)),
                       method=self.optimizer)
        self.l, self.sigma_f = res.x

    def predict(self, X_test, return_std=False):
        K = self.rbf_kernel(self.X_train, self.X_train, self.l, self.sigma_f) + \
            self.alpha * np.eye(len(self.X_train))
        K_s = self.rbf_kernel(self.X_train, X_test, self.l, self.sigma_f)
        K_ss = self.rbf_kernel(X_test, X_test, self.l, self.sigma_f) + 1e-8 * np.eye(len(X_test))
        K_inv = np.linalg.inv(K)
        
        # Posterior mean
        mu_s = K_s.T.dot(K_inv).dot(self.y_train)
        
        if return_std:
            # Posterior standard deviation
            sigma_s = np.sqrt(np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s)))
            return mu_s, sigma_s
        return mu_s