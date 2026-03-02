"""
Aether Bayesian Kernel - Performance Analytics Engine
Calculates institutional-grade risk and return attribution metrics, 
providing the mathematical basis for strategy validation and audit reporting.
"""

import pandas as pd
import numpy as np

class StrategyValidator:
    """
    Coordinates the statistical attribution of backtest results.
    
    Transforms raw equity time-series into annualized risk-adjusted metrics, 
    evaluating the stability and quality of the generated alpha.
    """

    @staticmethod
    def calculate_metrics(equity_series, dates):
        """
        Executes a comprehensive statistical audit of an equity curve.
        
        Args:
            equity_series (array-like): Historical capital liquidation values.
            dates (Index): Temporal axis corresponding to the equity values.
            
        Returns:
            dict: Performance metrics including Sharpe, Sortino, and Max Drawdown.
        """
        df = pd.Series(equity_series, index=dates)
        returns = df.pct_change().dropna()
        
        # --- PHASE I: RISK-ADJUSTED ATTRIBUTION ---
        # Annualized performance metrics utilizing standard 252-day factors
        total_ret = (df.iloc[-1] / df.iloc[0]) - 1
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Downside-deviation focused risk measurement
        downside_rets = returns[returns < 0]
        sortino = (returns.mean() / downside_rets.std()) * np.sqrt(252) if not downside_rets.empty else 0
        
        # --- PHASE II: STABILITY AND VOLATILITY ---
        # Peak-to-trough erosion analysis
        cum_max = df.cummax()
        drawdown = (df - cum_max) / (cum_max + 1e-6)
        
        vol = returns.std() * np.sqrt(252)
        
        return {
            "Return": f"{total_ret * 100:.2f}%",
            "Sharpe": round(float(sharpe), 2),
            "Sortino": round(float(sortino), 2),
            "MaxDD": f"{drawdown.min() * 100:.2f}%",
            "Volatility": f"{vol * 100:.2f}%",
            "Final": df.iloc[-1],
            "DrawdownSeries": drawdown 
        }

    def print_terminal_report(self, ticker, m_l, m_u, local_ccy):
        """
        Renders a high-density audit memorandum to the system console.
        
        Args:
            ticker (str): Asset identifier.
            m_l (dict): Metrics calculated on the local currency basis.
            m_u (dict): Metrics calculated on the USD base currency.
            local_ccy (str): Identifier for the local currency.
        """
        width = 65
        print("\n" + "="*width)
        print(f" STRATEGY PERFORMANCE AUDIT: {ticker.upper()}")
        print("-" * width)
        print(f"{'METRIC':<20} | {'LOCAL ('+local_ccy+')':<18} | {'BASE (USD)':<15}")
        print("-" * width)
        
        # Mapping metrics to standardized reporting fields
        stats = [
            ("Total Return", m_l['Return'], m_u['Return']),
            ("Sharpe Ratio", str(m_l['Sharpe']), str(m_u['Sharpe'])),
            ("Max Drawdown", m_l['MaxDD'], m_u['MaxDD']),
            ("Final Liquidation", f"{local_ccy} {m_l['Final']:,.0f}", f"$ {m_u['Final']:,.0f}")
        ]
        
        for name, l, u in stats:
            print(f"{name:<20} | {l:<18} | {u:<15}")
            
        print("="*width + "\n")