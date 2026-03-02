"""
Aether Bayesian Kernel - Relative Strength Engine
Specialist engine designed for cross-sectional alpha discovery. Ranks global 
universes based on risk-adjusted velocity to isolate leadership and 
enforce portfolio concentration in high-conviction assets.
"""

import pandas as pd
import numpy as np
from src.strategies.base import Strategy

class RelativeStrengthEngine(Strategy):
    """
    Coordinates global rotation and leadership selection.
    
    Utilizes a tournament-style ranking system to allocate capital 
    strictly to assets exhibiting superior reward-to-risk characteristics 
    relative to the broader universe.
    """

    def __init__(self, top_n=1):
        """
        Initializes the rotation specialist.
        
        Args:
            top_n (int): The number of leading assets to select for allocation.
        """
        super().__init__("Relative_Strength")
        self.top_n = top_n
        
    def generate_signals(self, market_data_slice, regime_probs):
        """
        Estimates relative conviction scores across the current universe.
        
        Args:
            market_data_slice (dict): Current standardized market data.
            regime_probs (np.ndarray): Recursive Bayesian state probabilities.
            
        Returns:
            dict: Mapping of ticker identifiers to ordinal conviction weights.
        """
        # --- PHASE I: SYSTEMIC RISK FILTRATION ---
        # Strategy enters a defensive 'Halt' state if the Crash regime (State 3) 
        # probability exceeds the 50% critical threshold.
        if regime_probs[3] > 0.5:
            return {}
            
        scores = {}
        
        # --- PHASE II: RISK-ADJUSTED VELOCITY SCORING ---
        # Evaluates the quality of directional momentum by consuming the 
        # proprietary 'v' (Velocity) sense from the standardized data stream.
        for symbol, row in market_data_slice.items():
            if 'v' in row:
                # Velocity sense already represents Return / Risk normalization
                scores[symbol] = row['v']
        
        if not scores:
            return {}
            
        # --- PHASE III: ORDINAL SELECTION ---
        # Performs a cross-sectional sort to isolate the top-performing 
        # quintile or specific N-leaders as defined in the mandate.
        sorted_assets = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_assets = [asset[0] for asset in sorted_assets[:self.top_n]]
        
        signals = {}
        for symbol in scores.keys():
            if symbol in top_assets:
                # High conviction assigned to statistical leaders
                signals[symbol] = 1.0 
            else:
                # Disqualifies laggards to prevent capital dilution
                signals[symbol] = 0.0 
                
        return signals