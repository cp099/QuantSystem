import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

class RegimeDetector:
    def __init__(self, n_components=4):
        """
        n_components=4 corresponds to:
        1. Low Vol / Bull
        2. High Vol / Bear
        3. Range / Sideways
        4. Transition / Shock
        """
        self.model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.feature_cols = ['vol_ratio', 'momentum', 'vol_short'] # The core drivers
        
    def fit(self, features):
        """
        Trains the GMM on historical data.
        """
        X = features[self.feature_cols].values
        self.model.fit(X)
        print(f"Model trained. Converged: {self.model.converged_}")
        return self

    def predict(self, features):
        """
        Returns the regime label (0-3) for each timestamp.
        """
        X = features[self.feature_cols].values
        return self.model.predict(X)
    
    def predict_proba(self, features):
        """
        Returns probabilities of each regime.
        Crucial for 'Soft' switching (e.g., 60% Bullish, 40% Transition).
        """
        X = features[self.feature_cols].values
        return self.model.predict_proba(X)

    def save(self, filepath="src/gmm_model.pkl"):
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath="src/gmm_model.pkl"):
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")