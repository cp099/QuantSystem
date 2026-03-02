"""
Aether Bayesian Kernel - Trend Persistence Engine
Specialist engine designed to capture persistent directional moves during 
expansionary Bayesian regimes (Low-Volatility Bull and Recovery dynamics).
"""

import pandas as pd
import numpy as np
from src.strategies.base import Strategy

class TrendEngine(Strategy):
    """
    Coordinates directional participation in trending environments.
    
    Utilizes Bayesian state belief to confirm market expansion and validates 
    asset-level strength through realized momentum persistence.
    """

    def __init__(self):
        """
        Initializes the trend persistence specialist.
        """
        super().__init__("Trend_Engine")
        
    def generate_signals(self, market_data_slice, regime_probs):
        """
        Estimates directional conviction across the global universe.
        
        Args:
            market_data_slice (dict): Current standardized market data.
            regime_probs (np.ndarray): Recursive Bayesian state probabilities.
            
        Returns:
            dict: Mapping of ticker identifiers to trend conviction weights.
        """
        signals = {}
        
        # --- PHASE I: STATE-SPACE GATING ---
        # Strategy only activates when expansionary dynamics (States 0 or 4) 
        # represent a cumulative probability exceeding 50%.
        bullish_prob = regime_probs[0] + regime_probs[4]
        
        if bullish_prob < 0.5:
            return {} 
            
        # --- PHASE II: DIRECTIONAL PERSISTENCE ANALYSIS ---
        # Scans the universe for assets exhibiting positive realized momentum 
        # within the confirmed expansionary regime.
        for symbol, row in market_data_slice.items():
            if 'Close' not in row or 'Momentum' not in row:
                continue
            
            # Logic requires positive absolute momentum for long participation.
            if row['Momentum'] > 0:
                # Sizing is deferred to the Institutional Risk Engine.
                signals[symbol] = 1.0
            else:
                signals[symbol] = 0.0
                
        return signals