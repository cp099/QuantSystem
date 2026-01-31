from src.strategies.base import Strategy

class VolatilityBreakoutEngine(Strategy):
    """
    V2 Volatility Breakout:
    - Enters when Volatility expands rapidly from a low base.
    - Uses 'Vol_ratio' feature (Short Vol / Long Vol).
    """
    def __init__(self):
        super().__init__("Vol_Breakout")
        
    def generate_signals(self, market_data_slice, regime_probs):
        signals = {}
        
        # 1. Regime Check
        favorable = regime_probs[0] + regime_probs[1] + regime_probs[4]
        if favorable < 0.3:
            return {}

        for symbol, row in market_data_slice.items():
            # Logic:
            # 1. Squeeze: We assume prior days had low vol (we can't see history here easily without state,
            #    but Vol_ratio < 1.0 implies compression).
            # 2. Expansion: Current Normalized_return is massive (> 2.0 sigma)
            
            if 'Vol_ratio' in row and 'Normalized_return' in row:
                if row['Vol_ratio'] < 0.9 and row['Normalized_return'] > 2.0:
                    signals[symbol] = 1.0
                else:
                    signals[symbol] = 0.0
                    
        return signals