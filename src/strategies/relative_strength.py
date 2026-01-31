import pandas as pd
from src.strategies.base import Strategy

class RelativeStrengthEngine(Strategy):
    """
    Global Rotation Strategy:
    - Ranks all assets by volatility-adjusted momentum.
    - Buys only the top N assets (Concentration).
    - Active in almost all regimes except Crash (State 3).
    """
    def __init__(self, top_n=1):
        super().__init__("Relative_Strength")
        self.top_n = top_n
        
    def generate_signals(self, market_data_slice, regime_probs):
        # 1. Safety Check: If Crash Regime (State 3) is dominant, stay out.
        if regime_probs[3] > 0.5:
            return {}
            
        scores = {}
        
        # 2. Score Assets
        for symbol, row in market_data_slice.items():
            if 'Momentum' in row and 'Atr' in row and row['Atr'] > 0:
                vol_pct = row['Atr'] / row['Close']
                score = row['Momentum'] / vol_pct
                scores[symbol] = score
        
        if not scores:
            return {}
            
        # 3. Rank and Select Top N
        sorted_assets = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_assets = [asset[0] for asset in sorted_assets[:self.top_n]]
        
        signals = {}
        for symbol in scores.keys():
            if symbol in top_assets:
                signals[symbol] = 1.0 # Buy Top N
            else:
                signals[symbol] = 0.0 # Ignore laggards
                
        return signals