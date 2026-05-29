"""
Aether Bayesian Kernel - Deep Learning Strategy Verification
Validates the functional and mathematical integrity of the DeepLearningEngine.
"""

import sys
import os
import pandas as pd
import numpy as np

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.deep_learning import DeepLearningEngine

def test_deep_learning_signals():
    """
    Verifies that DeepLearningEngine generates valid probability conviction signals.
    """
    print("[TEST] INITIATING DEEP LEARNING ENGINE AUDIT...")
    
    # Instantiate engine (loads models/global_neural_network.pkl)
    engine = DeepLearningEngine()
    
    # Verify model is successfully loaded
    assert engine.model is not None, "Global Neural Network model failed to load."
    
    # Mock standardized feature market slice
    market_slice = {
        'AAPL': pd.Series({'v': 1.5, 'r': 0.5, 'c': -0.2, 'a': 1.0, 'd': 0.8, 'l': 0.1, 'b': 0.9}),
        'MSFT': pd.Series({'v': -0.5, 'r': 1.2, 'c': 0.5, 'a': -0.8, 'd': 1.1, 'l': 0.1, 'b': 0.4}),
        'GLD': pd.Series({'v': np.nan, 'r': 0.5, 'c': -0.2, 'a': 1.0, 'd': 0.8, 'l': 0.1, 'b': 0.9})  # Has NaN
    }
    
    # Simulating standard regime probabilities (unused by DL but required by interface)
    regime_probs = np.array([0.5, 0.1, 0.2, 0.1, 0.1])
    
    sigs = engine.generate_signals(market_slice, regime_probs)
    print("\n[DL Test] Generated Convictions:")
    print(sigs)
    
    # Verify return type and keys
    assert isinstance(sigs, dict)
    assert 'AAPL' in sigs
    assert 'MSFT' in sigs
    assert 'GLD' in sigs
    
    # Verify probability limits
    assert 0.0 <= sigs['AAPL'] <= 1.0
    assert 0.0 <= sigs['MSFT'] <= 1.0
    
    # Verify NaN failsafe (should default to 0.0)
    assert sigs['GLD'] == 0.0
    
    print("[TEST] DEEP LEARNING ENGINE AUDIT COMPLETE.")
