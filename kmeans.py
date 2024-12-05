import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to clusters
            distances = self._calc_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for i in range(self.k):
                self.centroids[i] = np.mean(X[labels == i], axis=0)
            
            # Check for convergence
            if np.all(np.abs(old_centroids - self.centroids) < self.tol):
                break

    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)

    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances

    def inertia(self, X):
        labels = self.predict(X)
        return sum(np.min(self._calc_distances(X), axis=1))
