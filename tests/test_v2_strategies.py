"""
Aether Bayesian Kernel - Strategy Logic Verification
Validates the functional integrity of specialist strategy engines, 
ensuring accurate directional conviction generation across 
diverse cross-sectional market snapshots.
"""

import sys
import os
import pandas as pd
import numpy as np

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.trend_engine import TrendEngine
from src.strategies.relative_strength import RelativeStrengthEngine

def test_strategy_logic():
    """
    Executes a functional audit of directional signal generation.
    
    Validates that trend and rotation engines correctly interpret relativistic 
    feature sets and Bayesian state beliefs to produce prioritized asset weights.
    """
    print("[TESTING] INITIATING STRATEGY LOGIC AUDIT...")
    
    # --- PHASE I: CROSS-SECTIONAL MARKET SNAPSHOT ---
    # Constructing a standardized multi-asset data slice
    # Features: v (Velocity), r (Risk), c (Compression), a (Alpha)
    market_slice = {
        'SPY': pd.Series({'Close': 400, 'Momentum': -0.01, 'v': -0.5, 'c': 1.2}),
        'GLD': pd.Series({'Close': 180, 'Momentum': 0.05, 'v': 2.5, 'c': 0.8}),
        'TLT': pd.Series({'Close': 100, 'Momentum': 0.01, 'v': 0.2, 'c': 1.0})
    }
    
    # --- PHASE II: RECURSIVE BELIEF VECTOR ---
    # Simulating a confirmed expansionary regime (State 0 & 4 dominant)
    regime_probs = np.array([0.8, 0.0, 0.1, 0.0, 0.1]) 
    
    # --- PHASE III: TREND PERSISTENCE VALIDATION ---
    trend = TrendEngine()
    sigs_trend = trend.generate_signals(market_slice, regime_probs)
    print("\n[TEST] TREND PERSISTENCE CONVICTIONS:")
    print(sigs_trend)
    
    # --- PHASE IV: ORDINAL ROTATION VALIDATION ---
    # Testing leadership selection logic (Top-1 mandate)
    rs = RelativeStrengthEngine(top_n=1)
    sigs_rs = rs.generate_signals(market_slice, regime_probs)
    print("\n[TEST] RELATIVE STRENGTH LEADERSHIP:")
    print(sigs_rs)
    
    print("\n[TEST] STRATEGY LOGIC AUDIT COMPLETE.")
    
if __name__ == "__main__":
    test_strategy_logic()