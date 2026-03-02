"""
Aether Bayesian Kernel - Offline Statistical Anchor
Implements the Hidden Markov Model (HMM) framework to define baseline 
transition topologies and emission distributions for global market regimes.
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import joblib
import os

class RegimeDetectorV2:
    """
    Coordinates the training of the foundational statistical model.
    
    Utilizes an Expectation-Maximization (EM) algorithm to estimate the 
    parameters of a Gaussian Hidden Markov Model across multi-asset time-series.
    """

    def __init__(self, config):
        """
        Initializes the HMM Kernel.
        
        Args:
            config (dict): System configuration containing regime and feature parameters.
        """
        self.config = config['regime_model']
        self.n_components = self.config['n_components']
        self.feature_cols = self.config['features']
        self.model_path = self.config['model_save_path']
        
        # Core HMM Configuration
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, 
            covariance_type="full", 
            n_iter=200,          
            random_state=42,
            verbose=True,        
            tol=0.001            
        )

    def fit(self, all_features_df):
        """
        Trains the anchor model on concatenated cross-sectional data.
        
        Args:
            all_features_df (pd.DataFrame): Concatenated features from the universe.
            
        Returns:
            self: The trained detector instance.
        """
        print("[SYSTEM ANCHOR] OPTIMIZING UNIVERSAL TRANSITION TOPOLOGY...")
        X = all_features_df[self.feature_cols].values
        
        # Sequence length tracking for cross-sectional alignment
        lengths = all_features_df.groupby('symbol').size().values
        
        self.model.fit(X, lengths)
        
        print(f"[SYSTEM ANCHOR] OPTIMIZATION CONVERGED: {self.model.monitor_.converged}")
        self.print_summary()
        self.save()
        return self

    def predict(self, features_df):
        """
        Estimates the most likely hidden state sequence using the Viterbi algorithm.
        """
        X = features_df[self.feature_cols].values
        return self.model.predict(X)

    def save(self):
        """Persists the trained anchor state to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"[SYSTEM ANCHOR] STATE PERSISTED: {self.model_path}")

    def load(self):
        """Restores the anchor state from persistence."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Anchor file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"[SYSTEM ANCHOR] STATE RESTORED: {self.model_path}")
        return self
        
    def print_summary(self):
        """
        Displays a summary of the learned statistical priors.
        """
        print("\n--- TRANSITION PROBABILITY MATRIX ---")
        df_trans = pd.DataFrame(self.model.transmat_, 
                                index=[f'State {i}' for i in range(self.n_components)],
                                columns=[f'State {i}' for i in range(self.n_components)])
        print(df_trans.round(3))
        
        print("\n--- EMISSION DISTRIBUTION MEANS ---")
        df_means = pd.DataFrame(self.model.means_, 
                                columns=self.feature_cols,
                                index=[f'State {i}' for i in range(self.n_components)])
        print(df_means.round(3))