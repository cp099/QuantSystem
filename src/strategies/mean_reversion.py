from src.strategies.base import Strategy
import pandas as pd

class MeanReversionEngine(Strategy):
    """
    Logic: Buy when price is significantly below the Moving Average (Oversold),
    but ONLY if we are in a Range Regime.
    """
    def __init__(self):
        super().__init__("Mean_Reversion")
        
    def generate_signal(self, current_bar, current_regime):
        """
        current_bar expects: 'Close', 'Dist_ma' (Distance from MA), 'RSI' (optional)
        """
        # 1. Regime Filter: Only trade in Regime 3 (Chop) or Regime 2 (Recovery)
        # Avoid Regime 0 (Strong Trend - don't short tops) and Regime 1 (Crash - don't catch knives)
        if current_regime not in [2, 3]:
            return 0
            
        # 2. Entry Logic
        # 'Dist_ma' comes from FeatureEngineer. E.g., -0.05 means 5% below MA.
        # Buy if we are extended downwards (-2% deviation)
        if current_bar['Dist_ma'] < -0.02:
            return 1.0 # Long
            
        # 3. Exit Logic (Simple Mean Reversion)
        # Sell if we return to the mean (Dist_ma > 0)
        elif current_bar['Dist_ma'] > 0:
            return 0 # Flat
            
        return 0 # No Action