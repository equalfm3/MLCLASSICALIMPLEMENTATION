import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting

    def fit(self, X, y):
        for name, estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.voting == 'hard':
            predictions = np.asarray([estimator.predict(X) for _, estimator in self.estimators]).T
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
            return maj
        elif self.voting == 'soft':
            predictions = self._predict_proba(X)
            return np.argmax(predictions, axis=1)

    def _predict_proba(self, X):
        probas = [estimator.predict_proba(X) for _, estimator in self.estimators]
        return np.average(probas, axis=0)

    def predict_proba(self, X):
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when voting='hard'")
        return self._predict_proba(X)