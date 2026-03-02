import yfinance as yf
import pandas as pd
import numpy as np

class UniversalLoader:
    def get_market_context(self, ticker):
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            return "^NSEI", "INR", "USDINR=X", ["RELIANCE.NS", "SBIN.NS", "TCS.NS"]
        return "SPY", "USD", None, ["AAPL", "MSFT", "NVDA"]

    def fetch_and_engineer(self, ticker):
        bench, local_ccy, fx_ticker, breadth_basket = self.get_market_context(ticker)
        print(f">>>> Robust-Syncing: {ticker}")
        
        all_tickers = [ticker, bench, "^TNX", "^IRX"] + breadth_basket
        if fx_ticker: all_tickers.append(fx_ticker)
        
        data = yf.download(all_tickers, period="7y", auto_adjust=True, progress=False)
        def get_df(t): return data.xs(t, axis=1, level=1).ffill().bfill()

        df = get_df(ticker)
        df_bench = get_df(bench)
        df_tnx = get_df("^TNX"); df_irx = get_df("^IRX")
        
        # 1. RAW FEATURE EXTRACTION
        atr = (df['High'] - df['Low']).rolling(14).mean()
        df['v_raw'] = (df['Close'] - df['Close'].shift(1)) / (atr + 1e-6)
        df['r_raw'] = atr / (df['Close'] + 1e-6)
        
        rets = df['Close'].pct_change()
        df['vol_pct'] = rets.rolling(20).std()
        df['c_raw'] = rets.rolling(20).std() / (rets.rolling(100).std() + 1e-6)
        
        # Alpha: 20-day spread vs Index (Stay in decimals)
        df['a_raw'] = rets.rolling(20).sum() - df_bench['Close'].pct_change(20)
        
        df['d_raw'] = rets.rolling(20).std() / (df_bench['Close'].pct_change(20).std() + 1e-6)
        df['l_raw'] = (df_tnx['Close'] - df_irx['Close']) / 10.0
        
        breadth_counts = []
        for t in breadth_basket:
            try:
                px = get_df(t)['Close']
                breadth_counts.append((px > px.rolling(200).mean()).astype(int))
            except: continue
        df['b_raw'] = pd.concat(breadth_counts, axis=1).mean(axis=1)

        # 2. ROBUST CLAMPING LAYER (The "Iron Clad" Fix)
        cols_to_scale = ['v_raw', 'r_raw', 'c_raw', 'a_raw', 'd_raw', 'l_raw', 'b_raw']
        for col in cols_to_scale:
            window = df[col].rolling(500)
            median = window.median()
            iqr = window.quantile(0.75) - window.quantile(0.25)
            # Standardize and CLAMP to [-4, 4] to prevent numerical overflow
            z = (df[col] - median) / (iqr + 1e-6)
            df[col.replace('_raw', '')] = np.clip(z, -4.0, 4.0)

        df['FX_Rate'] = get_df(fx_ticker)['Close'].reindex(df.index).ffill() if fx_ticker else 1.0
            
        return df.dropna(), df_bench.reindex(df.index).ffill(), local_ccy