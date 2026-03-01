import pandas as pd
import numpy as np
from hmmlearn import hmm
import joblib
import os

class RegimeDetectorV2:
    def __init__(self, config):
        """Initializes the HMM detector from a configuration dictionary."""
        self.config = config['regime_model']
        self.n_components = self.config['n_components']
        self.feature_cols = self.config['features']
        self.model_path = self.config['model_save_path']
        
        # Using GaussianHMM for continuous features
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, 
            covariance_type="full", 
            n_iter=200,          # More iterations for a more complex model
            random_state=42,
            verbose=True,        # Show convergence progress
            tol=0.001            # Stricter tolerance
        )

    def fit(self, all_features_df):
        """
        Trains the HMM on a concatenated DataFrame of features from ALL assets.
        This is crucial for learning universal, asset-agnostic market dynamics.
        """
        print("--- Training HMM on Universal Feature Set ---")
        # The model expects a single numpy array of shape (n_samples, n_features)
        X = all_features_df[self.feature_cols].values
        
        # The model also needs an array of lengths to know where each asset's sequence ends
        lengths = all_features_df.groupby('symbol').size().values
        
        self.model.fit(X, lengths)
        
        print(f"\nHMM Model converged: {self.model.monitor_.converged}")
        self.print_summary()
        self.save()
        return self

    def predict(self, features_df):
        """Predicts the most likely sequence of regimes for a given asset."""
        X = features_df[self.feature_cols].values
        return self.model.predict(X)

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"\nRegime model V2 saved to {self.model_path}")

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}.")
        self.model = joblib.load(self.model_path)
        print(f"Regime model V2 loaded from {self.model_path}")
        return self
        
    def print_summary(self):
        """Prints the learned transition matrix and mean feature values for each state."""
        print("\n--- HMM Analysis ---")
        print("\nTransition Matrix (from row -> to col):")
        # Rows are "from state", columns are "to state"
        df_trans = pd.DataFrame(self.model.transmat_, 
                                index=[f'State {i}' for i in range(self.n_components)],
                                columns=[f'State {i}' for i in range(self.n_components)])
        print(df_trans.round(3))
        
        print("\nMean Feature Values per State:")
        df_means = pd.DataFrame(self.model.means_, 
                                columns=self.feature_cols,
                                index=[f'State {i}' for i in range(self.n_components)])
        print(df_means.round(3))