from src.strategies.base import Strategy
import pandas as pd

class MeanReversionEngine(Strategy):
    """
    V2 Mean Reversion Engine:
    - Thrives in State 2 (Grind/Range).
    - Buys when Normalized Return is sharply negative (Oversold).
    """
    def __init__(self):
        super().__init__("Mean_Reversion")
        
    def generate_signals(self, market_data_slice, regime_probs):
        signals = {}
        
        # 1. Regime Check (Favor State 2 - Grind)
        range_prob = regime_probs[2]
        
        if range_prob < 0.4:
            return {}
            
        # 2. Asset Check
        for symbol, row in market_data_slice.items():
            if 'Normalized_return' in row and row['Normalized_return'] < -1.5:
                signals[symbol] = 1.0
            else:
                signals[symbol] = 0.0
                
        return signals