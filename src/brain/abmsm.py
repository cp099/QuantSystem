import numpy as np
from scipy.stats import multivariate_normal
import joblib
import os

class ABMSM:
    def __init__(self, K, D, alpha=0.01, lam=0.999):
        self.K = K
        self.D = D
        self.alpha = alpha
        self.lam = lam
        self.pi = np.ones(K) / K
        self.A = np.eye(K)
        self.C = np.ones((K, K)) * 10.0
        self.means = np.zeros((K, D))
        self.covs = np.array([np.eye(D) for _ in range(K)])

    @staticmethod
    def load(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found at {filepath}")
        return joblib.load(filepath)

    def update(self, x_t):
        x_t = np.array(x_t).flatten()
        # 1. Predict
        pi_prior = self.A.T @ self.pi
        # 2. Likelihood
        L = np.zeros(self.K)
        for k in range(self.K):
            stabilized_cov = self.covs[k] + np.eye(self.D) * 1e-6
            L[k] = multivariate_normal.pdf(x_t, mean=self.means[k], cov=stabilized_cov)
        # 3. Update
        pi_post = L * pi_prior
        norm = np.sum(pi_post)
        self.pi = pi_post / norm if norm > 1e-12 else pi_prior
        # 4. Adapt
        self._adapt(x_t, L)
        return self.pi

    def _adapt(self, x_t, L):
        numerator = self.pi[:, np.newaxis] * self.A * L[np.newaxis, :]
        xi = numerator / (np.sum(numerator) + 1e-12)
        self.C = (self.lam * self.C) + xi
        self.A = self.C / (self.C.sum(axis=1)[:, np.newaxis] + 1e-12)
        for k in range(self.K):
            eta = self.alpha * self.pi[k]
            diff = x_t - self.means[k]
            self.means[k] += eta * diff
            self.covs[k] = (1 - eta) * self.covs[k] + eta * np.outer(diff, diff)

    def get_entropy(self):
        p = self.pi + 1e-9
        return -np.sum(p * np.log2(p)) / np.log2(self.K)