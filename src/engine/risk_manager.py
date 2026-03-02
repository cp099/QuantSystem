import numpy as np

class RiskManager:
    def __init__(self, target_vol=0.12): # Lowered to 12% for Iron Clad safety
        self.target_vol = target_vol / np.sqrt(252)

    def calculate_position_size(self, equity, asset_vol, signal, current_dd):
        """
        Institutional 'Defensive' Sizing:
        1. Vol-Targeting Base
        2. Signal Weighting
        3. Drawdown Penalty (The Veto)
        """
        if asset_vol <= 0 or np.isnan(asset_vol): return 0
        
        # 1. Base Size
        base_size = (equity * self.target_vol) / asset_vol
        
        # 2. Drawdown Penalty
        # If DD is -10%, we reduce size by 40%. If DD is -20%, we reduce by 80%.
        dd_penalty = max(0, 1.0 - (abs(current_dd) * 4.0))
        
        # 3. Final Calculation
        final_size = base_size * signal * dd_penalty
        
        # Cap at 1.0x Leverage (Institutional Safety)
        return min(final_size, equity)