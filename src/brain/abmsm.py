import numpy as np
from scipy.stats import multivariate_normal
import joblib

class ABMSM:
    def __init__(self, K=5, D=5, alpha=0.02, lam=0.99):
        self.K, self.D = K, D
        self.alpha, self.lam = alpha, lam
        self.pi = np.ones(K) / K
        self.A = np.eye(K) * 0.9 + 0.1 / K
        self.C = self.A * 50.0
        self.means = np.zeros((K, D))
        self.covs = np.array([np.eye(D) for _ in range(K)])

    def update(self, x_t, adapt=True):
        x_t = np.array(x_t).flatten()
        pi_prior = self.A.T @ self.pi
        L = np.zeros(self.K)
        for k in range(self.K):
            reg_cov = self.covs[k] + np.eye(self.D) * 1e-3
            L[k] = multivariate_normal.pdf(x_t, mean=self.means[k], cov=reg_cov)
        pi_post = L * pi_prior
        norm = np.sum(pi_post)
        self.pi = pi_post / norm if norm > 1e-12 else pi_prior
        if adapt: self._adapt(x_t, L)
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

    def get_bull_states(self):
        # Bull = High Velocity (0) + High Relative Alpha (3)
        return np.argsort([m[0] + m[3] for m in self.means])[-2:]

    def get_entropy(self):
        p = self.pi + 1e-9
        return -np.sum(p * np.log2(p)) / np.log2(self.K)

    @staticmethod
    def load(filepath): return joblib.load(filepath)