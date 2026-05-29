"""
PROJECT: AETHER BAYESIAN KERNEL (ABK)
MODULE:  CORE BRAIN (ABMSM)
VERSION: 3.5.0
DESCRIPTION:
    Implements a public open-source entry point for the ABMSM engine.
    If the proprietary implementation is available locally, it delegates execution.
    Otherwise, it runs a baseline simulated fallback algorithm to allow 
    public visual testing on the terminal dashboard.
"""

import os
import sys
import numpy as np

# Attempt to load the proprietary/secret implementation
try:
    from src.brain.abmsm_secret import ABMSM as SecretABMSM
except ImportError:
    SecretABMSM = None

class ABMSM:
    """
    Adaptive Bayesian Markov-Switching Engine.
    Public wrapper shell. Integrates secret proprietary kernel when present.
    """
    def __init__(self, K=5, D=7, alpha=0.1, lam=0.9):
        if SecretABMSM is not None:
            super().__setattr__('_impl', SecretABMSM(K, D, alpha, lam))
            self.K = self._impl.K
            self.D = self._impl.D
            self.pi = self._impl.pi
            self.A = self._impl.A
            self.means = self._impl.means
            self.covs = self._impl.covs
        else:
            super().__setattr__('_impl', None)
            self.K, self.D = K, D
            self.alpha, self.lam = alpha, lam
            self.pi = np.ones(K) / K
            self.A = np.eye(K) * 0.6 + 0.4 / K
            self.means = np.zeros((K, D))
            self.covs = np.array([np.eye(D) * 1.0 for _ in range(K)])

    def update(self, x_t, adapt=True):
        impl = self.__dict__.get('_impl')
        if impl is not None:
            res = impl.update(x_t, adapt)
            # Sync variables for external observers
            self.pi = impl.pi
            self.A = impl.A
            self.means = impl.means
            self.covs = impl.covs
            return res
        
        # --- Public Fallback: Simulated Regime Update ---
        x_t = np.array(x_t).flatten()
        if np.isnan(x_t).any():
            return self.pi
        
        v = x_t[0] if len(x_t) > 0 else 0.0
        r = x_t[1] if len(x_t) > 1 else 0.0
        
        # Compute generic scores representing fake regimes
        scores = np.zeros(self.K)
        scores[0] = max(0, v)            # Growth (Low Vol Up)
        scores[1] = max(0, -v) * r       # Crash (High Vol Down)
        scores[2] = max(0, 1 - abs(v))   # Mean Reverting
        scores[3] = max(0, abs(v) * r)   # Volatility Breakout
        scores[4] = max(0, 0.5 - r)      # Sideways Consolidation
        
        scores_sum = scores.sum()
        if scores_sum > 0:
            scores /= scores_sum
        else:
            scores = np.ones(self.K) / self.K
            
        self.pi = 0.95 * self.pi + 0.05 * scores
        self.pi /= self.pi.sum()
        return self.pi

    def get_bull_states(self):
        impl = self.__dict__.get('_impl')
        if impl is not None:
            return impl.get_bull_states()
        return [0, 3] # Default mock index for Growth and Breakout

    def get_bayesian_signal(self):
        impl = self.__dict__.get('_impl')
        if impl is not None:
            return impl.get_bayesian_signal()
        bull_idx = self.get_bull_states()
        p_bull = sum(self.pi[i] for i in bull_idx)
        return p_bull * (1.0 - (self.get_entropy() * 0.7))

    def get_entropy(self):
        impl = self.__dict__.get('_impl')
        if impl is not None:
            return impl.get_entropy()
        p = np.clip(self.pi, 1e-9, 1.0)
        return -np.sum(p * np.log2(p)) / np.log2(self.K)

    def __getattr__(self, name):
        if name == '_impl':
            raise AttributeError()
        impl = self.__dict__.get('_impl')
        if impl is not None:
            return getattr(impl, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == '_impl':
            super().__setattr__(name, value)
            return
        impl = self.__dict__.get('_impl')
        if impl is not None:
            setattr(impl, name, value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def load(filepath):
        import joblib
        return joblib.load(filepath)