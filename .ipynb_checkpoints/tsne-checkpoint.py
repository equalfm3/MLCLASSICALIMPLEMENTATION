import numpy as np
from sklearn.metrics import pairwise_distances

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit_transform(self, X):
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        distances = pairwise_distances(X, metric='euclidean')
        
        # Compute joint probabilities p_ij
        P = self._compute_joint_probabilities(distances, self.perplexity)
        
        # Initialize low-dimensional embeddings
        np.random.seed(self.random_state)
        Y = np.random.randn(n_samples, self.n_components) * 0.0001
        
        # Early exaggeration
        P *= 4.0
        
        # Initialize past values for momentum
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()
        
        # Optimize
        for i in range(self.n_iter):
            # Compute low-dimensional affinities q_ij
            Q = self._compute_q_affinities(Y)
            
            # Compute gradients
            grads = self._compute_gradients(P, Q, Y)
            
            # Apply momentum and update learning rate
            momentum = 0.5 if i < 250 else 0.8
            learning_rate = self.learning_rate
            
            # Update embeddings
            Y_new = Y - learning_rate * grads + momentum * (Y - Y_m1) + (momentum**2) * (Y_m1 - Y_m2)
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()
            Y = Y_new
            
            # Zero-center embeddings
            Y -= np.mean(Y, axis=0)
            
            if i == 100:
                P /= 4.0  # Remove early exaggeration
            
            if (i + 1) % 100 == 0:
                cost = self._compute_kl_divergence(P, Q)
                print(f"Iteration {i+1}/{self.n_iter}, KL divergence: {cost:.4f}")
        
        return Y

    def _compute_joint_probabilities(self, distances, perplexity):
        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            P[i, :] = self._binary_search_perplexity(distances[i], perplexity)
        
        P = (P + P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-12)
        
        return P

    def _binary_search_perplexity(self, distances, perplexity):
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf
        
        for _ in range(50):
            sum_P = np.sum(np.exp(-distances * beta))
            if sum_P == 0:
                H = 0
            else:
                H = np.log(sum_P) + beta * np.sum(distances * np.exp(-distances * beta)) / sum_P
            
            Hdiff = H - np.log(perplexity)
            
            if np.abs(Hdiff) < 1e-5:
                break
            
            if Hdiff > 0:
                beta_min = beta
                beta = (beta + beta_max) / 2 if beta_max != np.inf else beta * 2
            else:
                beta_max = beta
                beta = (beta + beta_min) / 2 if beta_min != -np.inf else beta / 2
        
        P = np.exp(-distances * beta)
        P[np.isnan(P)] = 0
        P[distances == 0] = 0  # Set self-probabilities to 0
        return P / np.sum(P)

    def _compute_q_affinities(self, Y):
        distances = pairwise_distances(Y, metric='euclidean')
        inv_distances = 1 / (1 + distances**2)
        np.fill_diagonal(inv_distances, 0)
        return inv_distances / np.sum(inv_distances)

    def _compute_gradients(self, P, Q, Y):
        pq_diff = P - Q
        Y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        distances = pairwise_distances(Y, metric='euclidean')
        inv_distances = 1 / (1 + distances**2)
        np.fill_diagonal(inv_distances, 0)
        return 4 * (np.expand_dims(pq_diff, 2) * inv_distances[:, :, np.newaxis] * Y_diff).sum(axis=1)

    def _compute_kl_divergence(self, P, Q):
        return np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12)))