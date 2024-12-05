import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit(self, X, y, method='batch'):
        if method == 'batch':
            return self._batch_gradient_descent(X, y)
        elif method == 'stochastic':
            return self._stochastic_gradient_descent(X, y)
        elif method == 'mini_batch':
            return self._mini_batch_gradient_descent(X, y)
        else:
            raise ValueError("Method must be 'batch', 'stochastic', or 'mini_batch'")

    def _batch_gradient_descent(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.cost_history = []

        for i in range(self.max_iterations):
            h = np.dot(X, self.theta)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.learning_rate * gradient
            
            cost = np.sum((h - y) ** 2) / (2 * m)
            self.cost_history.append(cost)
            
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break

        return self.theta, self.cost_history

    def _stochastic_gradient_descent(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.cost_history = []

        for i in range(self.max_iterations):
            for j in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                h = np.dot(xi, self.theta)
                gradient = np.dot(xi.T, (h - yi))
                self.theta -= self.learning_rate * gradient

            h = np.dot(X, self.theta)
            cost = np.sum((h - y) ** 2) / (2 * m)
            self.cost_history.append(cost)
            
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break

        return self.theta, self.cost_history

    def _mini_batch_gradient_descent(self, X, y, batch_size=32):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.cost_history = []

        for i in range(self.max_iterations):
            indices = np.random.permutation(m)
            X = X[indices]
            y = y[indices]
            
            for j in range(0, m, batch_size):
                xi = X[j:j+batch_size]
                yi = y[j:j+batch_size]
                h = np.dot(xi, self.theta)
                gradient = np.dot(xi.T, (h - yi)) / batch_size
                self.theta -= self.learning_rate * gradient

            h = np.dot(X, self.theta)
            cost = np.sum((h - y) ** 2) / (2 * m)
            self.cost_history.append(cost)
            
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break

        return self.theta, self.cost_history

    def predict(self, X):
        return np.dot(X, self.theta)