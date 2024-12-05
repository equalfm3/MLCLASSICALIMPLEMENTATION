import numpy as np
from numpy.random import RandomState

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.height = 0

    def fit(self, X, current_height, rng):
        self.size = X.shape[0]
        if current_height >= self.height_limit or self.size <= 1:
            self.height = current_height
            return
        
        self.split_feature = rng.randint(X.shape[1])
        min_x = X[:, self.split_feature].min()
        max_x = X[:, self.split_feature].max()
        
        if min_x == max_x:
            self.height = current_height
            return
        
        self.split_value = rng.uniform(min_x, max_x)
        
        left_indices = X[:, self.split_feature] < self.split_value
        X_left = X[left_indices]
        X_right = X[~left_indices]
        
        self.left = IsolationTree(self.height_limit)
        self.right = IsolationTree(self.height_limit)
        self.left.fit(X_left, current_height + 1, rng)
        self.right.fit(X_right, current_height + 1, rng)
        self.height = max(self.left.height, self.right.height)

    def path_length(self, x, current_height):
        if self.split_feature is None:
            return current_height
        
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)

class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256, contamination=0.1, random_state=None):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self.rng = RandomState(random_state)
        self.trees = []
        self.threshold = None

    def fit(self, X):
        self.trees = []
        height_limit = int(np.ceil(np.log2(self.sample_size)))
        
        for _ in range(self.n_trees):
            sample_indices = self.rng.choice(X.shape[0], self.sample_size, replace=False)
            X_sample = X[sample_indices]
            tree = IsolationTree(height_limit)
            tree.fit(X_sample, 0, self.rng)
            self.trees.append(tree)
        
        path_lengths = self.compute_path_lengths(X)
        self.threshold = np.percentile(path_lengths, 100 * (1 - self.contamination))
        return self

    def compute_path_lengths(self, X):
        path_lengths = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            path_lengths[i] = np.mean([tree.path_length(x, 0) for tree in self.trees])
        return path_lengths

    def predict(self, X):
        path_lengths = self.compute_path_lengths(X)
        return np.where(path_lengths >= self.threshold, 1, -1)

    def decision_function(self, X):
        path_lengths = self.compute_path_lengths(X)
        return self.threshold - path_lengths