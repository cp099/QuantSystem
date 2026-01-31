import pandas as pd
import numpy as np
from src.strategies.base import Strategy

class TrendEngine(Strategy):
    """
    V2 Trend Engine:
    - Iterates through all assets.
    - Checks Regime (Favor State 0 & 4).
    - Checks Asset-Specific Trend (Price > MA).
    """
    def __init__(self):
        super().__init__("Trend_Engine")
        
    def generate_signals(self, market_data_slice, regime_probs):
        signals = {}
        
        # 1. Regime Check
        bullish_prob = regime_probs[0] + regime_probs[4]
        
        if bullish_prob < 0.5:
            return {} 
            
        # 2. Asset Check
        for symbol, row in market_data_slice.items():
            if 'Close' not in row or 'Momentum' not in row:
                continue
            
            if row['Momentum'] > 0:
                signals[symbol] = 1.0
            else:
                signals[symbol] = 0.0
                
        return signals