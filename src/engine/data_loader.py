import yfinance as yf
import pandas as pd
import numpy as np

class UniversalLoader:
    def get_market_context(self, ticker):
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            return "^NSEI", "INR", "USDINR=X"
        return "SPY", "USD", None

    def fetch_and_engineer(self, ticker):
        bench, local_ccy, fx_ticker = self.get_market_context(ticker)
        print(f">>> Syncing Global Feed: {ticker} | Benchmark: {bench}")
        
        tickers = [ticker, bench]
        if fx_ticker: tickers.append(fx_ticker)
        
        data = yf.download(tickers, period="5y", auto_adjust=True, progress=False)
        
        df_target = data.xs(ticker, axis=1, level=1).dropna()
        df_bench = data.xs(bench, axis=1, level=1).dropna()
        df_fx = data.xs(fx_ticker, axis=1, level=1) if fx_ticker else None

        df_target.columns = [c.capitalize() for c in df_target.columns]
        
        # --- PROPRIETARY SENSES (V3.5 - 5 DIMENSIONS) ---
        atr = (df_target['High'] - df_target['Low']).rolling(14).mean()
        
        # 1. Velocity
        df_target['Velocity'] = (df_target['Close'] - df_target['Close'].shift(1)) / atr
        # 2. Rel_Risk
        df_target['Rel_Risk'] = atr / atr.rolling(200).mean()
        # 3. Compression
        rets = df_target['Close'].pct_change()
        df_target['Compression'] = rets.rolling(20).std() / rets.rolling(100).std()
        # 4. Relative Alpha
        df_target['Rel_Alpha'] = rets.rolling(20).sum() - df_bench['Close'].pct_change().rolling(20).sum()
        # 5. Volatility Divergence (Stock Vol / Index Vol)
        df_target['Vol_Div'] = rets.rolling(20).std() / df_bench['Close'].pct_change().rolling(20).std()
        
        # FX Normalization
        if df_fx is not None:
            df_target['FX_Rate'] = df_fx['Close'].reindex(df_target.index).ffill()
        else:
            df_target['FX_Rate'] = 1.0
            
        return df_target.dropna(), df_bench.dropna(), local_ccy