import pandas as pd
import numpy as np

class StrategyValidator:
    @staticmethod
    def calculate_metrics(equity_series, dates):
        df = pd.Series(equity_series, index=dates)
        returns = df.pct_change().dropna()
        
        total_ret = (df.iloc[-1] / df.iloc[0]) - 1
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        sortino = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252) if not returns[returns < 0].empty else 0
        
        cum_max = df.cummax()
        drawdown = (df - cum_max) / (cum_max + 1e-6)
        
        return {
            "Return": f"{total_ret * 100:.2f}%",
            "Sharpe": round(float(sharpe), 2),
            "Sortino": round(float(sortino), 2),
            "MaxDD": f"{drawdown.min() * 100:.2f}%",
            "Volatility": f"{returns.std() * np.sqrt(252) * 100:.2f}%",
            "Final": df.iloc[-1],
            "DrawdownSeries": drawdown # CRITICAL KEY FOR PDF
        }

    def print_terminal_report(self, ticker, m_l, m_u, local_ccy):
        width = 65
        print("\n" + "="*width)
        print(f" INSTITUTIONAL STRATEGY AUDIT: {ticker.upper()}")
        print("-" * width)
        print(f"{'METRIC':<20} | {'LOCAL ('+local_ccy+')':<18} | {'BASE (USD)':<15}")
        print("-" * width)
        stats = [
            ("Total Return", m_l['Return'], m_u['Return']),
            ("Sharpe Ratio", str(m_l['Sharpe']), str(m_u['Sharpe'])),
            ("Max Drawdown", m_l['MaxDD'], m_u['MaxDD']),
            ("Final Equity", f"{local_ccy} {m_l['Final']:,.0f}", f"$ {m_u['Final']:,.0f}")
        ]
        for name, l, u in stats:
            print(f"{name:<20} | {l:<18} | {u:<15}")
        print("="*width + "\n")