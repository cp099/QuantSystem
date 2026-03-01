import numpy as np
from scipy.stats import multivariate_normal
import joblib
import os

class ABMSM:
    """
    Adaptive Bayesian Markov-Switching Model (Proprietary).
    
    Implements the recursive online learning cycle:
    1. Prediction (Time Update)
    2. Update (Measurement Update)
    3. Adaptation (Online Parameter Learning)
    """
    
    def __init__(self, n_regimes, n_features, learning_rate=0.01, decay_factor=0.999):
        # Hyperparameters
        self.K = n_regimes
        self.D = n_features
        self.alpha = learning_rate      # Emission learning rate (alpha)
        self.lam = decay_factor         # Transition decay factor (lambda)
        
        # State Variables
        self.pi = np.ones(self.K) / self.K          # Belief vector (pi_t)
        self.A = np.eye(self.K)                     # Transition Matrix (A_t)
        self.C = np.ones((self.K, self.K)) * 10.0   # Concentration Matrix (C_t)
        
        # Emission Parameters (theta_t)
        self.means = np.zeros((self.K, self.D))     # mu_{k,t}
        self.covs = np.array([np.eye(self.D) for _ in range(self.K)]) # Sigma_{k,t}
        
        # Cache for smoothing
        self.pi_prev = self.pi.copy()
        self.L_current = np.zeros(self.K)

    def initialize_from_hmm(self, hmm_model):
        """
        Warm-start the ABMSM using the static offline HMM parameters.
        This bridges the gap between historical training and online adaptation.
        """
        self.means = hmm_model.means_
        self.covs = hmm_model.covars_
        self.A = hmm_model.transmat_
        base_weight = 100.0
        self.C = self.A * base_weight
        print("ABMSM initialized from offline HMM.")

    def run_step(self, x_t):
        """
        Executes one full cycle of the engine for a new observation x_t.
        Returns the updated belief vector pi_t.
        """
        x_t = np.array(x_t).flatten()
        
        # --- STEP 1: PREDICTION (Time Update) ---
        # pi_{t|t-1} = A^T * pi_{t-1}
        pi_predicted = self.A.T @ self.pi
        
        # --- STEP 2: UPDATE (Measurement Update) ---
        # Calculate Likelihoods L_t(k)
        L_t = np.zeros(self.K)
        for k in range(self.K):
            stabilized_cov = self.covs[k] + np.eye(self.D) * 1e-6
            try:
                L_t[k] = multivariate_normal.pdf(x_t, mean=self.means[k], cov=stabilized_cov)
            except:
                L_t[k] = 1e-10
        
        self.L_current = L_t
        self.pi_prev = self.pi.copy() 
        
        pi_update = L_t * pi_predicted
        
        # Normalize: pi_t = pi' / sum(pi')
        norm_factor = np.sum(pi_update)
        if norm_factor < 1e-12:
            self.pi = pi_predicted 
        else:
            self.pi = pi_update / norm_factor
            
        # --- STEP 3: ADAPTATION (Online Learning) ---
        self._adapt_parameters(x_t)
        
        return self.pi

    def _adapt_parameters(self, x_t):
        """
        Internal method to update A, mu, and Sigma based on new belief.
        """
        # --- A. Transition Matrix Adaptation ---
        # Calculate Smoothed Posterior: xi_t(i, j)
        # Numerator: pi_{t-1}(i) * A(i,j) * L_t(j)
        
        # shape (K, 1) * shape (K, K) * shape (1, K) -> shape (K, K)
        numerator = self.pi_prev[:, np.newaxis] * self.A * self.L_current[np.newaxis, :]
        
        denom = np.sum(numerator)
        if denom > 1e-12:
            xi_t = numerator / denom
            
            # Concentration Update: C_t = lambda * C_{t-1} + xi_t
            self.C = (self.lam * self.C) + xi_t
            
            # Transition Matrix Update: Normalize rows of C
            row_sums = self.C.sum(axis=1)[:, np.newaxis]
            self.A = self.C / row_sums

        # --- B. Emission Parameter Adaptation ---
        for k in range(self.K):
            # Effective learning rate: eta = alpha * pi_t(k)
            eta = self.alpha * self.pi[k]
            
            # Mean Update
            # mu_new = (1-eta)*mu_old + eta*x_t
            diff = x_t - self.means[k]
            self.means[k] = self.means[k] + eta * diff
            
            # Covariance Update
            # Sigma_new = (1-eta)*Sigma_old + eta*(x-mu)(x-mu)^T
            diff_outer = np.outer(diff, diff)
            self.covs[k] = (1 - eta) * self.covs[k] + eta * diff_outer

    def save(self, filepath):
        joblib.dump(self, filepath)

    def get_entropy(self):
        """
        Proprietary Metric: Calculates the Shannon Entropy of the current belief.
        Scale: 0.0 (Absolute Certainty) to 1.0 (Complete Confusion).
        High entropy triggers an automatic portfolio 'Safe Mode'.
        """
        # Small epsilon to avoid log(0)
        p = self.pi + 1e-9
        entropy = -np.sum(p * np.log2(p)) / np.log2(self.K)
        return float(entropy)
        
    @staticmethod
    def load(filepath):
        return joblib.load(filepath)