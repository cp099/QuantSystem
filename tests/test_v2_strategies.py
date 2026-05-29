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
from src.strategies.mean_reversion import MeanReversionEngine
from src.strategies.volatility_breakout import VolatilityBreakoutEngine

def test_strategy_logic():
    """
    Executes a functional audit of directional signal generation.
    
    Validates that trend, rotation, reversion, and breakout engines correctly 
    interpret feature sets and Bayesian state beliefs to produce signals.
    """
    print("[TESTING] INITIATING STRATEGY LOGIC AUDIT...")
    
    # --- PHASE I: CROSS-SECTIONAL MARKET SNAPSHOT ---
    # Mocking standard feature mappings engineered in V3.5.1
    market_slice = {
        'SPY': pd.Series({'Close': 400, 'Momentum': -0.01, 'v': -0.5, 'c': 1.2, 'Normalized_return': -0.5, 'Vol_ratio': 1.2}),
        'GLD': pd.Series({'Close': 180, 'Momentum': 0.05, 'v': 2.5, 'c': 0.8, 'Normalized_return': 2.5, 'Vol_ratio': 0.8}),
        'TLT': pd.Series({'Close': 100, 'Momentum': 0.01, 'v': 0.2, 'c': 1.0, 'Normalized_return': 0.2, 'Vol_ratio': 1.0})
    }
    
    # --- PHASE II: TREND PERSISTENCE VALIDATION ---
    # Simulating a confirmed expansionary regime (State 0 & 4 dominant)
    regime_bull = np.array([0.8, 0.0, 0.1, 0.0, 0.1]) 
    trend = TrendEngine()
    sigs_trend = trend.generate_signals(market_slice, regime_bull)
    print("\n[TEST] TREND PERSISTENCE CONVICTIONS:")
    print(sigs_trend)
    # Expected: TrendEngine signals if Close & Momentum > 0
    assert sigs_trend['GLD'] == 1.0
    assert sigs_trend['TLT'] == 1.0
    assert sigs_trend['SPY'] == 0.0
    
    # --- PHASE III: ORDINAL ROTATION VALIDATION ---
    # Testing leadership selection logic (Top-1 mandate)
    rs = RelativeStrengthEngine(top_n=1)
    sigs_rs = rs.generate_signals(market_slice, regime_bull)
    print("\n[TEST] RELATIVE STRENGTH LEADERSHIP:")
    print(sigs_rs)
    # Expected: Top 1 is GLD (v=2.5)
    assert sigs_rs['GLD'] == 1.0
    assert sigs_rs['SPY'] == 0.0
    assert sigs_rs['TLT'] == 0.0
    
    # --- PHASE IV: MEAN REVERSION VALIDATION ---
    # Simulating a stationary / range-bound regime (State 2 dominant)
    regime_mr = np.array([0.0, 0.0, 0.8, 0.0, 0.2])
    mr = MeanReversionEngine()
    # Update market slice to trigger oversold condition for SPY
    market_slice_mr = market_slice.copy()
    market_slice_mr['SPY'] = pd.Series({'Close': 400, 'Momentum': -0.05, 'v': -1.6, 'c': 1.2, 'Normalized_return': -1.6, 'Vol_ratio': 1.2})
    sigs_mr = mr.generate_signals(market_slice_mr, regime_mr)
    print("\n[TEST] MEAN REVERSION CONVICTIONS:")
    print(sigs_mr)
    # Expected: MR signals SPY oversold (< -1.5)
    assert sigs_mr['SPY'] == 1.0
    assert sigs_mr['GLD'] == 0.0
    assert sigs_mr['TLT'] == 0.0

    # --- PHASE V: VOLATILITY BREAKOUT VALIDATION ---
    # Simulating expansion shock (State 0, 1, 4 favorable)
    vol_break = VolatilityBreakoutEngine()
    sigs_vol = vol_break.generate_signals(market_slice, regime_bull)
    print("\n[TEST] VOLATILITY BREAKOUT CONVICTIONS:")
    print(sigs_vol)
    # Expected: Vol_ratio < 0.9 and Normalized_return > 2.0 (triggers on GLD)
    assert sigs_vol['GLD'] == 1.0
    assert sigs_vol['SPY'] == 0.0
    assert sigs_vol['TLT'] == 0.0
    
    print("\n[TEST] STRATEGY LOGIC AUDIT COMPLETE.")
    
if __name__ == "__main__":
    test_strategy_logic()