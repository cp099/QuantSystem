"""
Aether Bayesian Kernel - Capital Allocation Engine
Coordinates hierarchical capital distribution across strategy engines 
based on recursive Bayesian state belief and systemic risk concentration.
"""

import pandas as pd
import numpy as np
import yaml

class CapitalAllocator:
    """
    Manages regime-gated capital budgeting and signal aggregation.
    
    Transforms latent state probabilities into discrete capital budgets while 
    monitoring the eigenvalue dispersion of the correlation matrix to 
    penalize systemic risk clustering.
    """

    def __init__(self, config_path, engines):
        """
        Initializes the allocation kernel.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.engines = engines
        self.risk_params = self.config.get('risk', {})
        self.allocations = {}

    def calculate_correlation_penalty(self, history_df):
        """
        Detects risk concentration through Eigenvalue Decomposition.
        
        Analyzes the variance explained by the primary principal component of 
        the cross-sectional return matrix. If one factor dominates, the system 
        calculates a leverage penalty to preserve capital.
        
        Args:
            history_df (pd.DataFrame): Matrix of standardized returns.
            
        Returns:
            float: Deleveraging multiplier [0.0 - 1.0].
        """
        try:
            if history_df is None or history_df.empty or len(history_df.columns) < 2:
                return 1.0
            
            # Sanitization: Ensure data is numerically stable for decomposition
            clean_df = history_df.dropna().loc[:, (history_df.std() > 0)]
            if clean_df.shape[1] < 2: 
                return 1.0

            # --- PHASE I: SPECTRAL ANALYSIS ---
            corr_matrix = np.nan_to_num(clean_df.corr().values)
            eigenvals = np.linalg.eigvalsh(corr_matrix)
            
            # --- PHASE II: CONCENTRATION ATTRIBUTION ---
            max_eigen = np.max(eigenvals)
            total_var = np.sum(eigenvals)
            
            if total_var <= 0: 
                return 1.0
            
            # Ratio of variance explained by the lead eigenvector
            concentration = max_eigen / total_var
            thresh = self.risk_params.get('concentration_threshold', 0.7)
            
            # Linear deleveraging if concentration exceeds institutional threshold
            if concentration > thresh:
                return float(thresh / concentration)
            
            return 1.0
        except Exception:
            # Failsafe: Reverts to neutral multiplier on numerical instability
            return 1.0

    def update_allocations(self, regime_probs, market_slice, history_df):
        """
        Calculates target portfolio weights through multi-stage aggregation.
        """
        # --- STAGE I: REGIME-BASED BUDGETING ---
        p_bull = regime_probs[0] + regime_probs[4]
        p_chop = regime_probs[1] + regime_probs[2]
        
        engine_weights = {
            "Trend_Engine": p_bull * 0.5,
            "Relative_Strength": p_bull * 0.5,
            "Mean_Reversion": p_chop * 0.8,
            "Vol_Breakout": p_chop * 0.2
        }
        
        # --- STAGE II: SIGNAL AGGREGATION ---
        raw_asset_weights = {}
        for engine in self.engines:
            budget = engine_weights.get(engine.name, 0.0)
            if budget <= 0: 
                continue
            
            sigs = engine.generate_signals(market_slice, regime_probs)
            total_sig = sum(sigs.values())
            
            if total_sig > 0:
                for asset, weight in sigs.items():
                    raw_asset_weights[asset] = raw_asset_weights.get(asset, 0.0) + (weight / total_sig * budget)
        
        # --- STAGE III: SYSTEMIC RISK PENALTY ---
        penalty = self.calculate_correlation_penalty(history_df)
        
        # Final weights adjusted for risk concentration
        self.allocations = {k: v * penalty for k, v in raw_asset_weights.items()}
        return self.allocations