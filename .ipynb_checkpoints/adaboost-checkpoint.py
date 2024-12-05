import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(n_estimators)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        sample_weight = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=sample_weight)
            
            y_pred = estimator.predict(X)
            incorrect = y_pred != y
            
            estimator_error = np.sum(sample_weight * incorrect) / np.sum(sample_weight)
            
            if estimator_error >= 0.5:
                break
            
            estimator_weight = self.learning_rate * (np.log((1 - estimator_error) / estimator_error) + np.log(len(np.unique(y)) - 1))
            
            self.estimators_.append(estimator)
            self.estimator_weights_[i] = estimator_weight
            
            if estimator_error == 0:
                break
            
            sample_weight *= np.exp(estimator_weight * incorrect)
            sample_weight /= np.sum(sample_weight)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_[:len(self.estimators_)]])
        weighted_predictions = np.dot(self.estimator_weights_[:len(self.estimators_)], predictions)
        return np.sign(weighted_predictions)