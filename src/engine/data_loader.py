import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import os

class MultiAssetLoader:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.universe = self.config['data']['universe']
        self.base_currency = self.config['system']['base_currency']
        self.raw_data_path = "data/raw"
        self.processed_data_path = f"data/processed/{self.base_currency}"
        os.makedirs(self.processed_data_path, exist_ok=True)
        self.fx_rates = {}

    def _get_fx_rate(self, quote_ccy):
        if quote_ccy == self.base_currency: return None
        pair = f"{quote_ccy}{self.base_currency}=X"
        if pair not in self.fx_rates:
            print(f"Fetching FX rate for {pair}...")
            fx_df = yf.download(pair, period="5y", progress=False, auto_adjust=True)
            self.fx_rates[pair] = fx_df['Close']
        return self.fx_rates[pair]

    def load_and_process_data(self):
        print("\n--- Starting High-Density Data Processing ---")
        processed_data = {}
        
        for asset in self.universe:
            symbol = asset['symbol']
            currency = asset['currency']
            
            # Smart Search for the file in any subfolder of data/raw
            filepath = None
            for root, dirs, files in os.walk(self.raw_data_path):
                if f"{symbol}.parquet" in files:
                    filepath = os.path.join(root, f"{symbol}.parquet")
                    break
            
            if not filepath:
                print(f"FAILED to find raw data for {symbol}. Skipping.")
                continue
            
            print(f"Processing {symbol} (Found at {filepath})...")
            df = pd.read_parquet(filepath)
            
            # Clean columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [col.capitalize() for col in df.columns]
            df = df[~df.index.duplicated(keep='first')]
            
            # 1. FX CONVERSION
            if currency != self.base_currency:
                fx_rate = self._get_fx_rate(currency)
                if fx_rate is not None and not fx_rate.empty:
                    df_base = df.copy()
                    df_base, fx_rate = df_base.align(fx_rate, join='left', axis=0)
                    fx_rate = fx_rate.ffill().bfill()
                    for col in ['Open', 'High', 'Low', 'Close']:
                        p_vals = df_base[col].values.flatten()
                        f_vals = fx_rate.values.flatten()
                        df_base[col] = pd.Series(p_vals * f_vals, index=df_base.index)
                else: df_base = df.copy()
            else: df_base = df.copy()

            # 2. FEATURE ENGINEERING
            atr = (df_base['High'] - df_base['Low']).rolling(14).mean()
            df_base['Atr'] = atr
            df_base['Normalized_return'] = (df_base['Close'] - df_base['Close'].shift(1)) / atr
            df_base['Momentum'] = df_base['Close'].pct_change(periods=50) # The Engine needs this!
            
            vol_s = df_base['Close'].pct_change().rolling(20).std()
            vol_l = df_base['Close'].pct_change().rolling(100).std()
            df_base['Vol_ratio'] = vol_s / vol_l
            
            df_base.dropna(inplace=True)
            processed_data[symbol] = df_base
            
        print(f"--- Processing Complete. Assets Loaded: {list(processed_data.keys())} ---")
        return processed_data

if __name__ == "__main__":
    MultiAssetLoader().load_and_process_data()