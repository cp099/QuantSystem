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
        """Calculates risk concentration penalty using Eigenvalues."""
        try:
            if history_df.empty or len(history_df.columns) < 2:
                return 1.0
            
            # Clean data: Remove rows with NaNs and columns with zero variance
            clean_df = history_df.dropna().loc[:, (history_df.std() > 0)]
            if clean_df.shape[1] < 2: return 1.0

            # Compute Correlation
            corr_matrix = clean_df.corr().values
            
            # Clean Matrix: Fill any remaining NaNs or Infs with 0
            corr_matrix = np.nan_to_num(corr_matrix)

            # Eigenvalue Decomposition
            eigenvals = np.linalg.eigvalsh(corr_matrix)
            
            max_eigen = np.max(eigenvals)
            total_var = np.sum(eigenvals)
            
            if total_var <= 0: return 1.0
            
            concentration = max_eigen / total_var
            thresh = self.risk_params.get('concentration_threshold', 0.7)
            
            return thresh / concentration if concentration > thresh else 1.0
        except Exception:
            return 1.0 # Default to no penalty on math failure

    def update_allocations(self, regime_probs, market_slice, history_df=None):
        # 1. Map Regimes to Engine Budgets
        p_bull = regime_probs[0] + regime_probs[4]
        p_chop = regime_probs[1] + regime_probs[2]
        
        engine_weights = {
            "Trend_Engine": p_bull * 0.4,
            "Relative_Strength": p_bull * 0.6,
            "Mean_Reversion": p_chop * 0.8,
            "Vol_Breakout": p_chop * 0.2 + p_bull * 0.2
        }
        
        # 2. Collect Signals
        raw_weights = {}
        for engine in self.engines:
            budget = engine_weights.get(engine.name, 0.0)
            if budget < 0.01: continue
            
            sigs = engine.generate_signals(market_slice, regime_probs)
            total_sig = sum(sigs.values())
            if total_sig > 0:
                for asset, score in sigs.items():
                    raw_weights[asset] = raw_weights.get(asset, 0.0) + (score/total_sig * budget)

        # 3. Apply Correlation Penalty
        penalty = self.calculate_correlation_penalty(history_df) if history_df is not None else 1.0
        self.allocations = {k: v * penalty for k, v in raw_weights.items()}
        return self.allocations