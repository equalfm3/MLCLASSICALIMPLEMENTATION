import numpy as np

class XGBoostTree:
    def __init__(self, max_depth=3, learning_rate=0.1, min_child_weight=1):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.tree = {}

    def _calculate_gain(self, gradient, hessian):
        return np.square(gradient.sum()) / (hessian.sum() + self.min_child_weight)

    def _split(self, X, gradient, hessian):
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_gain = self._calculate_gain(gradient[left_mask], hessian[left_mask])
                right_gain = self._calculate_gain(gradient[right_mask], hessian[right_mask])
                gain = left_gain + right_gain - self._calculate_gain(gradient, hessian)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, gradient, hessian, depth=0):
        if depth >= self.max_depth or X.shape[0] <= 1:
            return -gradient.sum() / (hessian.sum() + self.min_child_weight)

        feature, threshold = self._split(X, gradient, hessian)
        if feature is None:
            return -gradient.sum() / (hessian.sum() + self.min_child_weight)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_branch = self._build_tree(X[left_mask], gradient[left_mask], hessian[left_mask], depth + 1)
        right_branch = self._build_tree(X[right_mask], gradient[right_mask], hessian[right_mask], depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_branch,
            'right': right_branch
        }

    def fit(self, X, gradient, hessian):
        self.tree = self._build_tree(X, gradient, hessian)

    def predict(self, X):
        def _traverse_tree(x, node):
            if isinstance(node, dict):
                if x[node['feature']] <= node['threshold']:
                    return _traverse_tree(x, node['left'])
                else:
                    return _traverse_tree(x, node['right'])
            else:
                return node

        return np.array([_traverse_tree(x, self.tree) for x in X])

class SimpleXGBoost:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.trees = []

    def _gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true)

    def _hessian(self, y_true, y_pred):
        return 2 * np.ones_like(y_true)

    def fit(self, X, y):
        y_pred = np.zeros_like(y)

        for _ in range(self.n_estimators):
            gradient = self._gradient(y, y_pred)
            hessian = self._hessian(y, y_pred)

            tree = XGBoostTree(max_depth=self.max_depth, learning_rate=self.learning_rate, min_child_weight=self.min_child_weight)
            tree.fit(X, gradient, hessian)
            
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred