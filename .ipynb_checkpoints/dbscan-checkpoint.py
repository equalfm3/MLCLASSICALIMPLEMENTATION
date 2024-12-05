import numpy as np
from collections import defaultdict

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        self.X = X
        self.labels_ = np.full(len(X), -1)
        self.cluster_id = 0
        
        for i in range(len(X)):
            if self.labels_[i] != -1:
                continue
            
            neighbors = self._find_neighbors(i)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
            else:
                self._expand_cluster(i, neighbors)
                self.cluster_id += 1
        
        return self.labels_

    def _find_neighbors(self, index):
        distances = np.linalg.norm(self.X - self.X[index], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, index, neighbors):
        self.labels_[index] = self.cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.labels_[neighbor] == -1:
                self.labels_[neighbor] = self.cluster_id
                
                new_neighbors = self._find_neighbors(neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])
            i += 1