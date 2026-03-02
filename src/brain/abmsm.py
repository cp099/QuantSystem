"""
PROJECT: AETHER BAYESIAN KERNEL (ABK)
MODULE:  CORE BRAIN (ABMSM)
VERSION: 3.5.0
AUTHOR:  [Chirag P Patil/cp099]
COPYRIGHT: (c) 2026 [Chirag P Patil/cp099]. ALL RIGHTS RESERVED.

CLASSIFICATION: PROPRIETARY & CONFIDENTIAL
DESCRIPTION: 
    Implements the Adaptive Bayesian Markov-Switching Model (ABMSM).
    This mathematical kernel performs online parameter adaptation
    utilizing Dirichlet-Multinomial transition memory and 
    Posterior-Weighted EWMA emission updates.

WARNING: Unauthorised distribution or reverse engineering of this 
mathematical model is strictly prohibited under Intellectual 
Property law.
"""

import numpy as np
from scipy.stats import multivariate_normal
import joblib

class ABMSM:
    """
    Adaptive Bayesian Markov-Switching Engine.
    
    Coordinates a recursive inference cycle to estimate latent market states 
    from a multi-dimensional relativistic feature space.
    """
    
    def __init__(self, K=5, D=7, alpha=0.1, lam=0.9):
        """
        Initializes the Bayesian Kernel.
        
        Args:
            K (int): Number of hidden regimes.
            D (int): Dimensionality of the feature vector.
            alpha (float): Online learning rate for emission parameters.
            lam (float): Forgetting factor for transition memory.
        """
        self.K, self.D = K, D
        self.alpha, self.lam = alpha, lam
        self.pi = np.ones(K) / K
        self.A = np.eye(K) * 0.6 + 0.4 / K 
        self.C = self.A * 10.0
        self.means = np.zeros((K, D))
        self.covs = np.array([np.eye(D) * 1.0 for _ in range(K)])

    def update(self, x_t, adapt=True):
        """
        Executes a singular recursive inference step.
        
        Performs time-update prediction and measurement-update correction 
        utilizing a log-space stabilized likelihood function.
        
        Args:
            x_t (array-like): Standardized 7-D feature vector.
            adapt (bool): Enables online parameter evolution.
            
        Returns:
            np.ndarray: Updated state belief vector (pi).
        """
        x_t = np.array(x_t).flatten()
        if np.isnan(x_t).any(): 
            return self.pi
        
        # --- State Prediction (Time Update) ---
        pi_prior = self.A.T @ self.pi
        
        # --- Likelihood Computation ---
        log_L = np.zeros(self.K)
        for k in range(self.K):
            # Static stabilization of the covariance matrix
            reg_cov = self.covs[k] + np.eye(self.D) * 0.5 
            try:
                log_L[k] = multivariate_normal.logpdf(x_t, mean=self.means[k], cov=reg_cov)
            except: 
                log_L[k] = -20.0
        
        # --- Posterior Estimation (Measurement Update) ---
        log_post = log_L + np.log(pi_prior + 1e-12)
        log_post -= np.max(log_post)
        pi_new = np.exp(log_post)
        
        # Prior Probability Injection (Minimum Doubt Floor)
        self.pi = (pi_new / np.sum(pi_new)) + 0.05
        self.pi /= self.pi.sum()
        
        if adapt: 
            self._adapt(x_t, np.exp(log_L))
        return self.pi

    def _adapt(self, x_t, L):
        """
        Calculates and applies recursive parameter updates.
        
        Implements the transition memory update and centroid migration 
        for each latent state distribution.
        """
        # Transition Memory Update
        numerator = self.pi[:, np.newaxis] * self.A * L[np.newaxis, :]
        xi = numerator / (np.sum(numerator) + 1e-12)
        self.C = (self.lam * self.C) + xi + 0.01
        self.A = self.C / (self.C.sum(axis=1)[:, np.newaxis] + 1e-12)
        
        # Distribution Centroid and Dispersion Migration
        for k in range(self.K):
            eta = self.alpha * self.pi[k]
            diff = x_t - self.means[k]
            self.means[k] += eta * diff
            self.covs[k] = (1 - eta) * self.covs[k] + eta * np.outer(diff, diff) + (np.eye(self.D) * 1e-3)

    def get_bull_states(self):
        """
        Identifies optimal growth regimes.
        
        Sorts hidden states based on the synthesis of directional velocity 
        and relative alpha components.
        """
        scores = [m[0] + m[3] for m in self.means]
        return np.argsort(scores)[-2:]

    def get_bayesian_signal(self):
        """
        Generates a confidence-weighted directional signal.
        
        Applies a non-linear uncertainty penalty to the growth state probabilities 
        to ensure risk-aversion during state transitions.
        """
        bull_idx = self.get_bull_states()
        p_bull = sum(self.pi[i] for i in bull_idx)
        return p_bull * (1.0 - (self.get_entropy() * 0.7))

    def get_entropy(self):
        """
        Calculates Normalized Shannon Entropy.
        
        Quantifies the mathematical uncertainty of the current state belief.
        """
        p = np.clip(self.pi, 1e-9, 1.0)
        return -np.sum(p * np.log2(p)) / np.log2(self.K)

    @staticmethod
    def load(filepath): 
        """Restores kernel state from persistence."""
        return joblib.load(filepath)