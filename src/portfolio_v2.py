import pandas as pd
import numpy as np
import yaml

class PortfolioControllerV2:
    def __init__(self, config_path='config.yaml', engines=[]):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.engines = engines
        self.risk_params = self.config['risk']
        self.allocations = {} 

    def calculate_correlation_penalty(self, history_df):
        """
        Calculates a leverage penalty based on the concentration of risk 
        (Eigenvalues of the Correlation Matrix).
        
        Input: history_df containing columns like ('SPY', 'Normalized_return'), ...
        """
        # 1. Extract Normalized Returns for all assets
        
        if history_df.empty or len(history_df.columns) < 2:
            return 1.0 
            
        # 2. Compute Correlation Matrix
        corr_matrix = history_df.corr()
        
        # 3. Eigenvalue Decomposition
        eigenvals = np.linalg.eigvalsh(corr_matrix)
        
        # 4. Calculate Concentration Ratio (Max Eigenvalue / Sum of Eigenvalues)
        max_eigen = max(eigenvals)
        total_variance = sum(eigenvals)
        
        if total_variance == 0: return 1.0
        
        concentration_ratio = max_eigen / total_variance
        
        # 5. Apply Penalty Logic
        threshold = self.risk_params.get('concentration_threshold', 0.7)
        
        if concentration_ratio > threshold:
            penalty = threshold / concentration_ratio
            return penalty
        
        return 1.0 

    def update_allocations(self, regime_probs, market_data_slice, history_returns_df=None):
        """
        Master Allocation Logic.
        1. Determine Engine Weights based on Regime.
        2. Ask Engines for Asset Signals.
        3. Aggregate Signals into Asset Weights.
        4. Apply Correlation Penalty.
        """
        # --- A. Regime -> Engine Allocation ---
        p_bull = regime_probs[0] + regime_probs[4]
        p_chop = regime_probs[1] + regime_probs[2]
        p_crash = regime_probs[3]
        
        engine_weights = {
            "Trend_Engine": p_bull * 0.4,
            "Relative_Strength": p_bull * 0.6,
            "Mean_Reversion": p_chop * 0.8,
            "Vol_Breakout": p_chop * 0.2 + p_bull * 0.2
        }
        
        # --- B. Engine -> Asset Allocation ---
        raw_asset_weights = {}
        
        for engine in self.engines:
            budget = engine_weights.get(engine.name, 0.0)
            if budget < 0.01: continue
            
            signals = engine.generate_signals(market_data_slice, regime_probs)
            
            total_sig = sum(signals.values())
            if total_sig > 0:
                for asset, score in signals.items():
                    allocation = (score / total_sig) * budget
                    raw_asset_weights[asset] = raw_asset_weights.get(asset, 0.0) + allocation

        # --- C. Global Risk Scaling (Correlation Penalty) ---
        penalty_factor = 1.0
        if history_returns_df is not None:
            penalty_factor = self.calculate_correlation_penalty(history_returns_df)
            
        # Apply Penalty
        final_allocations = {k: v * penalty_factor for k, v in raw_asset_weights.items()}
        
        # Save
        self.allocations = final_allocations
        return self.allocations