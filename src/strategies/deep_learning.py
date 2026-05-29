"""
Aether Bayesian Kernel - Deep Learning Engine
Specialist engine utilizing a global MLP Neural Network to predict directional 
conviction probabilities across diverse asset classes.
"""

import os
import joblib
import numpy as np
from src.strategies.base import Strategy

class DeepLearningEngine(Strategy):
    """
    Coordinates deep learning predictions using scale-invariant feature inputs.
    """

    def __init__(self, model_path="models/global_neural_network.pkl"):
        """
        Initializes the neural network specialist.
        """
        super().__init__("Deep_Learning")
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Restores the pre-trained neural network state from disk."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"[DEEP LEARNING] Model restored successfully from {self.model_path}")
            except Exception as e:
                print(f"[DEEP LEARNING] Error loading model from {self.model_path}: {e}")
        else:
            print(f"[DEEP LEARNING] WARNING: Model file {self.model_path} not found. Running with disabled signals.")

    def generate_signals(self, market_data_slice, regime_probs):
        """
        Predicts positive return probability for each asset in the current period.

        Args:
            market_data_slice (dict): Current standardized market data.
            regime_probs (np.ndarray): Recursive Bayesian state probabilities.

        Returns:
            dict: Mapping of ticker identifiers to neural network conviction weights [0.0 - 1.0].
        """
        if self.model is None:
            # Re-try loading in case it was compiled post-initialization
            self._load_model()
            if self.model is None:
                return {}

        signals = {}
        features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']

        for symbol, row in market_data_slice.items():
            try:
                # 1. Feature extraction
                x = np.array([row[f] for f in features]).reshape(1, -1)
                
                # Check for NaNs
                if np.isnan(x).any():
                    signals[symbol] = 0.0
                    continue
                
                # 2. Probability prediction [Class 0 (Down), Class 1 (Up)]
                prob_up = self.model.predict_proba(x)[0][1]
                
                # Assign conviction weight
                signals[symbol] = float(prob_up)
            except Exception:
                # Failsafe: return neutral probability on feature issues
                signals[symbol] = 0.0

        return signals
