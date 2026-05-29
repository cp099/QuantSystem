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
        # Expected Shortfall target is approx 2.06x daily volatility under normal distribution
        self.target_es = self.target_vol * 2.06

    def calculate_position_size(self, equity, asset_vol, signal, current_dd, asset_returns=None):
        """
        Calculates the risk-adjusted capital allocation for a specific asset.
        
        Args:
            equity (float): Current total liquidation value.
            asset_vol (float): Realized percentage volatility of the asset.
            signal (float): Confidence-weighted Bayesian signal [0.0 - 1.0].
            current_dd (float): Current peak-to-trough account drawdown (decimal).
            asset_returns (array-like, optional): Historical daily returns to compute Expected Shortfall.
            
        Returns:
            float: Optimal position size in local currency units.
        """
        # Default to standard volatility targeting
        risk_metric = asset_vol
        target_risk = self.target_vol
        
        if asset_returns is not None and len(asset_returns) > 20:
            try:
                rets = np.array(asset_returns)
                # Value at Risk at 95% confidence
                var_95 = np.percentile(rets, 5)
                # Expected Shortfall at 95% confidence (average of worst 5% outcomes)
                worst_outcomes = rets[rets <= var_95]
                if len(worst_outcomes) > 0:
                    es_95 = np.mean(worst_outcomes)
                    if es_95 < 0:
                        risk_metric = abs(es_95)
                        target_risk = self.target_es
            except Exception:
                pass

        if risk_metric <= 0 or np.isnan(risk_metric): 
            return 0
        
        # --- PHASE I: RISK TARGETING (Volatility or Expected Shortfall) ---
        base_size = (equity * target_risk) / risk_metric
        
        # --- PHASE II: DRAWDOWN FEEDBACK MODULATION ---
        # Calculates a penalty coefficient to aggressively reduce exposure as losses mount.
        # Implements a linear deleveraging schedule based on historical equity erosion.
        dd_penalty = max(0, 1.0 - (abs(current_dd) * 4.0))
        
        # --- PHASE III: CONSTRAINT ENFORCEMENT ---
        # Synthesis of base risk budget, model conviction, and defensive feedback.
        final_size = base_size * signal * dd_penalty
        
        # Hard constraint: Enforces a non-leveraged mandate (1.0x Equity)
        return min(final_size, equity)