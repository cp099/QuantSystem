"""
Aether Bayesian Kernel - Mean Reversion Engine
Specialist engine designed to exploit short-term statistical exhaustion 
during stationary market regimes (sideways/grind dynamics).
"""

from src.strategies.base import Strategy
import pandas as pd

class MeanReversionEngine(Strategy):
    """
    Coordinates equilibrium-restoration trades in low-conviction environments.
    
    Identifies assets that have deviated significantly from their local 
    statistical mean while the global market remains in a range-bound state.
    """

    def __init__(self):
        """
        Initializes the mean reversion specialist.
        """
        super().__init__("Mean_Reversion")
        
    def generate_signals(self, market_data_slice, regime_probs):
        """
        Estimates equilibrium recovery conviction across the universe.
        
        Args:
            market_data_slice (dict): Current standardized market data.
            regime_probs (np.ndarray): Recursive Bayesian state probabilities.
            
        Returns:
            dict: Mapping of ticker identifiers to reversion conviction weights.
        """
        signals = {}
        
        # --- PHASE I: STATE-SPACE GATING ---
        # Strategy only activates when the 'Grind' regime (State 2) is 
        # dominant or statistically significant (>40%).
        range_prob = regime_probs[2]
        
        if range_prob < 0.4:
            return {}
            
        # --- PHASE II: STATISTICAL EXHAUSTION DETECTION ---
        # Scans the universe for assets exhibiting significant z-score 
        # extension relative to historical realized volatility.
        for symbol, row in market_data_slice.items():
            # Entry logic utilizes ATR-normalized returns to detect oversold conditions.
            if 'Normalized_return' in row and row['Normalized_return'] < -1.5:
                # Binary assignment; sizing is subsequently handled by the Risk Engine.
                signals[symbol] = 1.0
            else:
                signals[symbol] = 0.0
                
        return signals