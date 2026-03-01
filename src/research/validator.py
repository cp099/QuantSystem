import pandas as pd
import numpy as np

class StrategyValidator:
    """Institutional Metrics Engine - Headless."""
    
    @staticmethod
    def calculate_metrics(equity_series):
        if not equity_series:
            return {"Error": "No data"}
            
        df = pd.Series(equity_series)
        returns = df.pct_change().dropna()
        
        # Risk-Adjusted Metrics
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Drawdown
        cum_max = df.cummax()
        drawdown = (df - cum_max) / cum_max
        max_dd = drawdown.min()
        
        return {
            "Total Return": f"{((df.iloc[-1] / df.iloc[0]) - 1) * 100:.2f}%",
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Final Equity": f"${df.iloc[-1]:,.2f}"
        }

    @staticmethod
    def print_report(name, metrics):
        print(f"\n{'='*45}")
        print(f" PERFORMANCE AUDIT: {name}")
        print(f"{'-'*45}")
        for k, v in metrics.items():
            print(f"{k:<20}: {v}")
        print(f"{'='*45}\n")