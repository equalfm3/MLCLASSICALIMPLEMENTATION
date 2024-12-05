import numpy as np
from itertools import product

class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=3, scoring='accuracy'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, X, y):
        param_combinations = [dict(zip(self.param_grid.keys(), v)) for v in product(*self.param_grid.values())]
        
        best_score = -np.inf
        best_params = None
        cv_results = []
        
        for params in param_combinations:
            scores = []
            for train_index, val_index in self._cv_split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                estimator = self.estimator.set_params(**params)
                estimator.fit(X_train, y_train)
                
                if self.scoring == 'accuracy':
                    score = estimator.score(X_val, y_val)
                else:
                    raise ValueError("Only 'accuracy' scoring is currently supported.")
                
                scores.append(score)
            
            mean_score = np.mean(scores)
            cv_results.append({'params': params, 'mean_test_score': mean_score})
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = self.estimator.set_params(**best_params)
        self.best_estimator_.fit(X, y)
        self.cv_results_ = cv_results

    def _cv_split(self, X):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.cv, n_samples // self.cv, dtype=int)
        fold_sizes[:n_samples % self.cv] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop], np.concatenate([indices[:start], indices[stop:]])
            current = stop

    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        return self.best_estimator_.predict(X)