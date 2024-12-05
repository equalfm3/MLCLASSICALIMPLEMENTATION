import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.base import BaseEstimator, ClusterMixin

class HierarchicalClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, linkage_method='ward'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.labels_ = None
        self.linkage_matrix = None

    def fit(self, X, y=None):
        self.linkage_matrix = linkage(X, method=self.linkage_method)
        self.labels_ = self._get_cluster_labels(self.linkage_matrix, self.n_clusters)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    @staticmethod
    def _get_cluster_labels(linkage_matrix, n_clusters):
        n_samples = linkage_matrix.shape[0] + 1
        labels = np.arange(n_samples)
        for i in range(n_samples - n_clusters):
            cluster1, cluster2, _, _ = linkage_matrix[i]
            labels[labels == cluster2] = cluster1
        unique_clusters = np.unique(labels)
        return np.array([np.where(unique_clusters == label)[0][0] for label in labels])

    def plot_dendrogram(self, **kwargs):
        return dendrogram(self.linkage_matrix, **kwargs)