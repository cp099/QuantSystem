"""
Aether Bayesian Kernel - Risk Management Engine
Implements a multi-layered risk control framework utilizing volatility targeting 
and dynamic drawdown-based deleveraging.
"""

import numpy as np

class RiskManager:
    """
    Coordinates capital preservation through adaptive position sizing.
    
    Synthesizes directional Bayesian conviction with realized asset volatility 
    and account-level drawdown to calculate optimal capital exposure.
    """

    def __init__(self, target_vol=0.12):
        """
        Initializes the risk kernel with an annualized volatility target.
        
        Args:
            target_vol (float): Annualized portfolio volatility target (decimal).
        """
        self.target_vol = target_vol / np.sqrt(252)

    def calculate_position_size(self, equity, asset_vol, signal, current_dd):
        """
        Calculates the risk-adjusted capital allocation for a specific asset.
        
        Args:
            equity (float): Current total liquidation value.
            asset_vol (float): Realized percentage volatility of the asset.
            signal (float): Confidence-weighted Bayesian signal [0.0 - 1.0].
            current_dd (float): Current peak-to-trough account drawdown (decimal).
            
        Returns:
            float: Optimal position size in local currency units.
        """
        if asset_vol <= 0 or np.isnan(asset_vol): 
            return 0
        
        # --- PHASE I: VOLATILITY TARGETING ---
        # Determines base exposure required to equalize the asset's risk contribution
        base_size = (equity * self.target_vol) / asset_vol
        
        # --- PHASE II: DRAWDOWN FEEDBACK MODULATION ---
        # Calculates a penalty coefficient to aggressively reduce exposure as losses mount.
        # Implements a linear deleveraging schedule based on historical equity erosion.
        dd_penalty = max(0, 1.0 - (abs(current_dd) * 4.0))
        
        # --- PHASE III: CONSTRAINT ENFORCEMENT ---
        # Synthesis of base risk budget, model conviction, and defensive feedback.
        final_size = base_size * signal * dd_penalty
        
        # Hard constraint: Enforces a non-leveraged mandate (1.0x Equity)
        return min(final_size, equity)