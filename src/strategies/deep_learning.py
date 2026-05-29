"""
Aether Bayesian Kernel - Deep Learning Engine
Specialist engine utilizing a global MLP Neural Network to predict directional 
conviction probabilities across diverse asset classes.
Public wrapper shell. Integrates secret proprietary kernel when present.
"""

import os
import numpy as np
from src.strategies.base import Strategy

try:
    from src.strategies.deep_learning_secret import DeepLearningEngine as DeepLearningEngineSecret
except ImportError:
    DeepLearningEngineSecret = None

class DeepLearningEngine(Strategy):
    """
    Coordinates deep learning predictions using scale-invariant feature inputs.
    Public wrapper shell. Integrates secret proprietary kernel when present.
    """

    def __init__(self, model_path="models/global_neural_network.pkl"):
        super().__init__("Deep_Learning")
        if DeepLearningEngineSecret is not None:
            super().__setattr__('_impl', DeepLearningEngineSecret(model_path))
            self.model_path = self._impl.model_path
            self.model = self._impl.model
        else:
            super().__setattr__('_impl', None)
            self.model_path = model_path
            self.model = None

    def generate_signals(self, market_data_slice, regime_probs):
        impl = self.__dict__.get('_impl')
        if impl is not None:
            return impl.generate_signals(market_data_slice, regime_probs)

        # Public Fallback: returns mock conviction signals based on momentum and trend features
        signals = {}
        for symbol, row in market_data_slice.items():
            try:
                # Generate a realistic-looking sigmoid activation score between 0.45 and 0.55
                v = row.get('v', 0.0)
                a = row.get('a', 0.0)
                score = 1.0 / (1.0 + np.exp(-(v * 0.2 + a * 0.1)))
                signals[symbol] = float(np.clip(score, 0.0, 1.0))
            except Exception:
                signals[symbol] = 0.5
        return signals

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
