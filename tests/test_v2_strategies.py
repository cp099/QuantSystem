import sys
import os
import pandas as pd
import numpy as np

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.trend_engine import TrendEngine
from src.strategies.relative_strength import RelativeStrengthEngine

def test_strategies():
    print("--- Testing V2 Strategy Engines ---")
    
    # 1. Mock Data Slice (Dictionary of Series)
    market_slice = {
        'SPY': pd.Series({'Close': 400, 'Momentum': -0.01, 'Atr': 4.0, 'Normalized_return': -0.5, 'Vol_ratio': 1.2}),
        'GLD': pd.Series({'Close': 180, 'Momentum': 0.05, 'Atr': 1.5, 'Normalized_return': 2.5, 'Vol_ratio': 0.8}),
        'TLT': pd.Series({'Close': 100, 'Momentum': 0.01, 'Atr': 1.0, 'Normalized_return': 0.2, 'Vol_ratio': 1.0})
    }
    
    # 2. Mock Regime (Bullish)
    regime_probs = np.array([0.8, 0.0, 0.1, 0.0, 0.1]) 
    
    # 3. Test Trend Engine
    trend = TrendEngine()
    sigs = trend.generate_signals(market_slice, regime_probs)
    print(f"\nTrend Signals: {sigs}")
    
    # 4. Test Relative Strength Engine
    rs = RelativeStrengthEngine(top_n=1)
    sigs_rs = rs.generate_signals(market_slice, regime_probs)
    print(f"RS Signals: {sigs_rs}")
    
if __name__ == "__main__":
    test_strategies()