import numpy as np
from scipy.stats import multivariate_normal
import joblib

class ABMSM:
    def __init__(self, K=5, D=7, alpha=0.1, lam=0.9):
        self.K, self.D = K, D
        self.alpha, self.lam = alpha, lam
        self.pi = np.ones(K) / K
        self.A = np.eye(K) * 0.6 + 0.4 / K # High flexibility
        self.C = self.A * 10.0
        self.means = np.zeros((K, D))
        self.covs = np.array([np.eye(D) * 1.0 for _ in range(K)])

    def update(self, x_t, adapt=True):
        x_t = np.array(x_t).flatten()
        if np.isnan(x_t).any(): return self.pi
        
        pi_prior = self.A.T @ self.pi
        
        # Log-Likelihood
        log_L = np.zeros(self.K)
        for k in range(self.K):
            # FIXED: Massive stabilizer (0.5) to prevent probability collapse
            reg_cov = self.covs[k] + np.eye(self.D) * 0.5 
            try:
                log_L[k] = multivariate_normal.logpdf(x_t, mean=self.means[k], cov=reg_cov)
            except: log_L[k] = -20.0
        
        # Softmax with Temperature
        log_post = log_L + np.log(pi_prior + 1e-12)
        log_post -= np.max(log_post)
        pi_new = np.exp(log_post)
        
        # Strong Bayesian Floor (5% minimum doubt)
        self.pi = (pi_new / np.sum(pi_new)) + 0.05
        self.pi /= self.pi.sum()
        
        if adapt: self._adapt(x_t, np.exp(log_L))
        return self.pi

    def _adapt(self, x_t, L):
        numerator = self.pi[:, np.newaxis] * self.A * L[np.newaxis, :]
        xi = numerator / (np.sum(numerator) + 1e-12)
        self.C = (self.lam * self.C) + xi + 0.01
        self.A = self.C / (self.C.sum(axis=1)[:, np.newaxis] + 1e-12)
        for k in range(self.K):
            eta = self.alpha * self.pi[k]
            diff = x_t - self.means[k]
            self.means[k] += eta * diff
            # Robust covariance update
            self.covs[k] = (1 - eta) * self.covs[k] + eta * np.outer(diff, diff) + (np.eye(self.D) * 1e-3)

    def get_bull_states(self):
        # Velocity (0) + Relative Alpha (3)
        scores = [m[0] + m[3] for m in self.means]
        return np.argsort(scores)[-2:]

    def get_bayesian_signal(self):
        bull_idx = self.get_bull_states()
        p_bull = sum(self.pi[i] for i in bull_idx)
        # Linear certainty scaling
        return p_bull * (1.0 - (self.get_entropy() * 0.7))

    def get_entropy(self):
        p = np.clip(self.pi, 1e-9, 1.0)
        return -np.sum(p * np.log2(p)) / np.log2(self.K)

    @staticmethod
    def load(filepath): return joblib.load(filepath)