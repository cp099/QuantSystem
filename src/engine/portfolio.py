"""
Aether Bayesian Kernel - Capital Allocation Engine
Coordinates hierarchical capital distribution across strategy engines 
based on recursive Bayesian state belief and systemic risk concentration.
"""

import pandas as pd
import numpy as np
import yaml
from scipy.optimize import minimize

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
            clean_df = history_df.ffill().bfill().dropna().loc[:, (history_df.std() > 0)]
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
            "Trend_Engine": p_bull * 0.4,
            "Relative_Strength": p_bull * 0.3,
            "Deep_Learning": p_bull * 0.3 + p_chop * 0.3,
            "Mean_Reversion": p_chop * 0.5,
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
        
        # --- STAGE III: SYSTEMIC RISK PENALTY & BAYESIAN OPTIMIZATION ---
        penalty = self.calculate_correlation_penalty(history_df)
        
        active_assets = [asset for asset, weight in raw_asset_weights.items() if weight > 0]
        
        if history_df is not None and len(active_assets) >= 2:
            try:
                # Align historical returns for active assets
                clean_df = history_df[active_assets].ffill().bfill().dropna()
                
                if len(clean_df) > 5 and clean_df.shape[1] == len(active_assets):
                    Sigma = clean_df.cov().values
                    
                    if not np.isnan(Sigma).any():
                        N = len(active_assets)
                        u_arr = np.array([raw_asset_weights[a] for a in active_assets])
                        
                        # Optimization: maximize utility - lambda/2 * w.T * Sigma * w
                        lam = 1.5  # Risk aversion
                        total_budget = sum(raw_asset_weights.values())
                        
                        def loss(w):
                            return - (np.dot(w, u_arr) - 0.5 * lam * np.dot(w, np.dot(Sigma, w)))
                        
                        cons = ({'type': 'ineq', 'fun': lambda w: total_budget - np.sum(w)})
                        bounds = [(0.0, total_budget) for _ in range(N)]
                        w0 = np.ones(N) * (total_budget / N)
                        
                        res = minimize(loss, w0, bounds=bounds, constraints=cons, method='SLSQP')
                        if res.success:
                            opt_allocs = {active_assets[i]: float(res.x[i]) for i in range(N)}
                            for k in raw_asset_weights.keys():
                                if k not in opt_allocs:
                                    opt_allocs[k] = 0.0
                            self.allocations = {k: v * penalty for k, v in opt_allocs.items()}
                            return self.allocations
            except Exception as e:
                print(f"[PORTFOLIO OPTIMIZER] Optimization failed: {e}")
                
        # Default fallback (backward-compatible)
        self.allocations = {k: v * penalty for k, v in raw_asset_weights.items()}
        return self.allocations