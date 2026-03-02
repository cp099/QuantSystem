"""
Aether Bayesian Kernel - Universal Data Ingestion & Normalization
Coordinates global market data synchronization and transforms raw time-series 
into a unit-less, relativistic state-space for the adaptive kernel.
"""

import yfinance as yf
import pandas as pd
import numpy as np

class UniversalLoader:
    """
    Coordinates multi-market data ingestion and relativistic scaling.
    
    Handles currency synchronization, benchmark alignment, and robust 
    scaling to ensure mathematical scale-invariance across global markets.
    """

    def get_market_context(self, ticker):
        """
        Maps a target security to its local benchmark and liquidity proxies.
        """
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            return "^NSEI", "INR", "USDINR=X", ["RELIANCE.NS", "SBIN.NS", "TCS.NS"]
        return "SPY", "USD", None, ["AAPL", "MSFT", "NVDA"]

    def fetch_and_engineer(self, ticker):
        """
        Inhales raw market data and executes the state-space transformation.
        
        Args:
            ticker (str): Global market identifier.
            
        Returns:
            tuple: (Standardized DataFrame, Benchmark DataFrame, Currency String)
        """
        bench, local_ccy, fx_ticker, breadth_basket = self.get_market_context(ticker)
        print(f"[DATA KERNEL] SYNCHRONIZING STATE-SPACE: {ticker}")
        
        all_tickers = [ticker, bench, "^TNX", "^IRX"] + breadth_basket
        if fx_ticker: 
            all_tickers.append(fx_ticker)
        
        # High-capacity lookback to populate normalization windows
        data = yf.download(all_tickers, period="7y", auto_adjust=True, progress=False)
        
        def get_df(t): 
            return data.xs(t, axis=1, level=1).ffill().bfill()

        df = get_df(ticker)
        df_bench = get_df(bench)
        df_tnx = get_df("^TNX")
        df_irx = get_df("^IRX")
        
        df.columns = [c.capitalize() for c in df.columns]
        
        # --- PHASE I: SYNTHETIC SENSE GENERATION ---
        
        # Directional Velocity and Local Risk
        atr = (df['High'] - df['Low']).rolling(14).mean()
        df['v_raw'] = (df['Close'] - df['Close'].shift(1)) / (atr + 1e-6)
        df['r_raw'] = atr / (df['Close'] + 1e-6)
        
        # Intrinsic Volatility Dynamics
        returns = df['Close'].pct_change()
        df['vol_pct'] = returns.rolling(20).std()
        df['c_raw'] = returns.rolling(20).std() / (returns.rolling(100).std() + 1e-6)
        
        # Cross-Sectional Alpha and Distribution Divergence
        df['a_raw'] = returns.rolling(20).sum() - df_bench['Close'].pct_change(20)
        df['d_raw'] = returns.rolling(20).std() / (df_bench['Close'].pct_change(20).std() + 1e-6)
        
        # Macro Liquidity and Participation Proxies
        df['l_raw'] = (df_tnx['Close'] - df_irx['Close']) / 10.0
        
        breadth_counts = []
        for t in breadth_basket:
            try:
                px = get_df(t)['Close']
                breadth_counts.append((px > px.rolling(200).mean()).astype(int))
            except: 
                continue
        df['b_raw'] = pd.concat(breadth_counts, axis=1).mean(axis=1)

        # --- PHASE II: NON-LINEAR ROBUST NORMALIZATION ---
        
        # Transformation utilizes Inter-Quartile Range (IQR) to mitigate outlier distortion
        cols_to_scale = ['v_raw', 'r_raw', 'c_raw', 'a_raw', 'd_raw', 'l_raw', 'b_raw']
        for col in cols_to_scale:
            window = df[col].rolling(500)
            median = window.median()
            iqr = window.quantile(0.75) - window.quantile(0.25)
            
            # Standardization to median-centered deviation units
            z = (df[col] - median) / (iqr + 1e-6)
            
            # Constraint applied to ensure numerical stability in high-dimensional space
            df[col.replace('_raw', '')] = np.clip(z, -4.0, 4.0)

        # Currency Basis Synchronization
        df['FX_Rate'] = get_df(fx_ticker)['Close'].reindex(df.index).ffill() if fx_ticker else 1.0
            
        return df.dropna(), df_bench.reindex(df.index).ffill(), local_ccy