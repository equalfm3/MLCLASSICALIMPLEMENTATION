import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = []
        
        for c in self.classes:
            X_c = X[y == c]
            self.parameters.append({
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0),
                'prior': len(X_c) / len(X)
            })

    def _pdf(self, x, mean, var):
        return np.exp(-((x-mean)**2)/(2*var)) / np.sqrt(2*np.pi*var)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                prior = np.log(self.parameters[i]['prior'])
                posterior = np.sum(np.log(self._pdf(x, self.parameters[i]['mean'], self.parameters[i]['var'])))
                posterior = prior + posterior
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)