import pandas as pd
import numpy as np
import yaml

class PortfolioControllerV2:
    def __init__(self, config_path, engines):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.engines = engines
        self.allocations = {}

    def update_allocations(self, regime_probs, market_slice, history_df):
        # Identify current dominant state
        p_bull = regime_probs[0] + regime_probs[4]
        p_chop = regime_probs[1] + regime_probs[2]
        
        # Engine Weighting
        engine_weights = {
            "Trend_Engine": p_bull * 0.5,
            "Relative_Strength": p_bull * 0.5,
            "Mean_Reversion": p_chop * 0.8,
            "Vol_Breakout": p_chop * 0.2
        }
        
        final_weights = {}
        for engine in self.engines:
            budget = engine_weights.get(engine.name, 0.0)
            if budget <= 0: continue
            
            # Request signals from strategy
            sigs = engine.generate_signals(market_slice, regime_probs)
            total_sig = sum(sigs.values())
            
            if total_sig > 0:
                for asset, weight in sigs.items():
                    final_weights[asset] = final_weights.get(asset, 0.0) + (weight / total_sig * budget)
                    
        self.allocations = final_weights
        return self.allocations