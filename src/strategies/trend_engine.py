import pandas as pd
import numpy as np
from src.strategies.base import Strategy

class TrendEngine(Strategy):
    """
    Traffic Light Trend Strategy:
    - GREEN LIGHT (Regime 0): Aggressive Long
    - YELLOW LIGHT (Regime 2): Moderate Long
    - RED LIGHT (Regime 1 or 3): Cash / Flat
    """
    def __init__(self):
        super().__init__("Trend_Engine")
        
    def generate_signal(self, current_bar, current_regime):
        # Regime 0 = Low Vol Bull (Best for Trend)
        if current_regime == 0:
            return 1.0 # 100% Long
            
        # Regime 2 = Recovery (Good for Trend)
        elif current_regime == 2:
            return 1.0 # Long
            
        # Regime 1 (Crash) or 3 (Chop) -> No Trend Trading
        return 0.0