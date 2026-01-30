import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
import os

class RegimeDetector:
    def __init__(self, config):
        """
        Initializes the detector based on a configuration dictionary.
        """
        self.config = config['regime_model']
        self.n_components = self.config['n_components']
        self.feature_cols = self.config['features']
        self.model_path = self.config['model_save_path']
        
        self.model = GaussianMixture(n_components=self.n_components, 
                                     covariance_type='full', 
                                     random_state=42)
        
    def fit(self, features):
        """
        Trains the GMM on historical data and saves the model.
        """
        X = features[self.feature_cols].values
        self.model.fit(X)
        print(f"Model trained. Converged: {self.model.converged_}")
        self.save() # Auto-save after fitting
        return self

    def predict_proba(self, features):
        """
        Returns probabilities of each regime for new data.
        """
        X = features[self.feature_cols].values
        return self.model.predict_proba(X)

    def save(self):
        """Saves the trained model object to the path specified in config."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Regime model saved to {self.model_path}")

    def load(self):
        """Loads a pre-trained model from the path specified in config."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train a model first.")
        self.model = joblib.load(self.model_path)
        print(f"Regime model loaded from {self.model_path}")
        return self