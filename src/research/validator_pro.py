import numpy as np
import pandas as pd

class AdvancedValidator:
    @staticmethod
    def run_monte_carlo(equity_series, simulations=1000):
        """
        Institutional Stress Test:
        Shuffles the sequence of historical returns 1,000 times.
        If the model still survives most paths, it's 'Iron Clad'.
        """
        returns = pd.Series(equity_series).pct_change().dropna()
        if returns.empty: return {"Error": "No data"}
        
        final_values = []
        max_drawdowns = []

        for _ in range(simulations):
            # Bootstrap returns (Sampling with replacement)
            shuffled_rets = np.random.choice(returns, size=len(returns), replace=True)
            # Generate random path
            path = 100000 * (1 + shuffled_rets).cumprod()
            
            final_values.append(path[-1])
            cum_max = np.maximum.accumulate(path)
            max_drawdowns.append(np.min((path - cum_max) / (cum_max + 1e-6)))

        return {
            "MC_Median_Return": f"{(np.median(final_values)/100000 - 1)*100:.1f}%",
            "MC_Risk_of_Ruin": f"{(np.array(max_drawdowns) < -0.40).mean()*100:.1f}%", # Chance of -40% drop
            "MC_VaR_95": f"{np.percentile(max_drawdowns, 5)*100:.1f}%" # 95% Confidence DD
        }