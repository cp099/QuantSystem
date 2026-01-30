import pandas as pd

class PortfolioController:
    def __init__(self, risk_manager, engines):
        self.risk_manager = risk_manager
        self.engines = engines # List of strategy instances
        self.allocations = {} # Store current weights

    def update_allocations(self, regime_probs):
        """
        Dynamic Capital Allocation based on Market Regime.
        regime_probs: [P(Bull), P(Crash), P(Recovery), P(Chop)]
        """
        p_bull = regime_probs[0]
        p_crash = regime_probs[1]
        p_recovery = regime_probs[2]
        p_chop = regime_probs[3]
        
        # 1. Define Strategy Preferences
        # Trend Engine loves Bull (0) and Recovery (2)
        trend_weight = p_bull + (0.5 * p_recovery)
        
        # Mean Reversion loves Chop (3) and Recovery (2)
        mean_rev_weight = p_chop + (0.5 * p_recovery)
        
        # 2. Crash Penalty (Cash Raising)
        # If Crash prob is high, reduce TOTAL exposure
        equity_utilization = 1.0 - p_crash # If 90% crash prob, only use 10% equity
        
        # 3. Normalize Weights
        total_raw_weight = trend_weight + mean_rev_weight
        if total_raw_weight == 0:
            self.allocations = {eng.name: 0.0 for eng in self.engines}
            return

        for engine in self.engines:
            if engine.name == "Trend_Engine":
                raw_w = trend_weight
            elif engine.name == "Mean_Reversion":
                raw_w = mean_rev_weight
            else:
                raw_w = 0
                
            # Scale by equity utilization
            # Example: 50% Trend, 50% MeanRev, but 50% Cash due to crash risk
            final_w = (raw_w / total_raw_weight) * equity_utilization
            self.allocations[engine.name] = final_w
            
    def get_allocation(self, strategy_name):
        return self.allocations.get(strategy_name, 0.0)