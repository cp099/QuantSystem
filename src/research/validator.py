import pandas as pd
import numpy as np

class StrategyValidator:
    @staticmethod
    def calculate_metrics(equity_series):
        if not equity_series: return {"Error": "No data"}
        df = pd.Series(equity_series)
        returns = df.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        cum_max = df.cummax()
        max_dd = ((df - cum_max) / cum_max).min()
        
        return {
            "Return": f"{((df.iloc[-1] / df.iloc[0]) - 1) * 100:.2f}%",
            "Sharpe": round(sharpe, 2),
            "MaxDD": f"{max_dd * 100:.2f}%",
            "Final": df.iloc[-1]
        }

    def print_dual_report(self, name, local_metrics, usd_metrics, local_ccy):
        print(f"\n{'='*60}")
        print(f" INVESTMENT MEMORANDUM: {name}")
        print(f"{'-'*60}")
        print(f"{'METRIC':<15} | {'LOCAL ('+local_ccy+')':<18} | {'BASE (USD)':<15}")
        print(f"{'-'*60}")
        print(f"{'Total Return':<15} | {local_metrics['Return']:<18} | {usd_metrics['Return']:<15}")
        print(f"{'Sharpe Ratio':<15} | {local_metrics['Sharpe']:<18} | {usd_metrics['Sharpe']:<15}")
        print(f"{'Max Drawdown':<15} | {local_metrics['MaxDD']:<18} | {usd_metrics['MaxDD']:<15}")
        print(f"{'Final Equity':<15} | {local_ccy} {local_metrics['Final']:,.0f} | ${usd_metrics['Final']:,.0f}")
        print(f"{'='*60}\n")