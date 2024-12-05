import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

class CustomUMAP(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=15, n_components=2, learning_rate=1.0, n_epochs=200, random_state=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.embedding_ = None

    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Compute nearest neighbors
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        # Initialize embedding
        self.embedding_ = np.random.normal(scale=0.0001, size=(n_samples, self.n_components))

        # Simplistic optimization
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                for j in indices[i][1:]:  # Skip the first neighbor (self)
                    grad = 2 * (self.embedding_[i] - self.embedding_[j])
                    self.embedding_[i] -= self.learning_rate * grad / self.n_neighbors
                    self.embedding_[j] += self.learning_rate * grad / self.n_neighbors

        return self

    def transform(self, X):
        return self.embedding_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)