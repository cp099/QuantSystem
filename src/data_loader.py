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

    def _calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def download_raw_data(self):
        print("--- Starting Raw Data Download ---")
        for asset in self.universe:
            symbol, asset_class = asset['symbol'], asset['asset_class']
            path = f"{self.raw_data_path}/{asset_class}/{asset.get('country', 'global')}"
            os.makedirs(path, exist_ok=True)
            filepath = f"{path}/{symbol}.parquet"
            print(f"Downloading {symbol}...")
            df = yf.download(symbol, period="5y", auto_adjust=True, progress=False)
            if df.empty:
                print(f"Warning: No data for {symbol}. Skipping.")
                continue
            df.to_parquet(filepath)
            print(f"Saved raw data for {symbol} to {filepath}")

    def load_and_process_data(self):
        print("\n--- Loading and Processing Data ---")
        processed_data = {}
        
        for asset in self.universe:
            symbol, asset_class, currency = asset['symbol'], asset['asset_class'], asset['currency']
            path = f"{self.raw_data_path}/{asset_class}/{asset.get('country', 'global')}"
            filepath = f"{path}/{symbol}.parquet"
            
            if not os.path.exists(filepath):
                print(f"Error: Raw data for {symbol} not found.")
                continue
            
            df = pd.read_parquet(filepath)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [col.capitalize() for col in df.columns]
            
            df = df[~df.index.duplicated(keep='first')]
            print(f"Processing {symbol}...")
            
            # 1. FX CONVERSION
            if currency != self.base_currency:
                fx_rate = self._get_fx_rate(currency)
                if fx_rate is None or fx_rate.empty:
                    print(f"Error fetching FX for {currency}. Skipping asset.")
                    continue
                
                df_base = df.copy()
                df_base, fx_rate = df_base.align(fx_rate, join='left', axis=0)
                fx_rate = fx_rate.ffill().bfill()
                
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in df_base.columns:
                        price_values = df_base[col].values.flatten()
                        fx_values = fx_rate.values.flatten()
                        multiplied_values = price_values * fx_values
                        df_base[col] = pd.Series(multiplied_values, index=df_base.index)
            else:
                df_base = df.copy()

            # 2. VOLATILITY NORMALIZATION
            required_cols = {'Open', 'High', 'Low', 'Close'}
            if not required_cols.issubset(df_base.columns):
                print(f"Warning: Skipping {symbol} due to missing required columns for ATR calc.")
                continue
            
            atr = self._calculate_atr(df_base)
            df_base['Atr'] = atr
            df_base['Normalized_return'] = (df_base['Close'] - df_base['Close'].shift(1)) / atr
            
            df_base.dropna(inplace=True)
            
            processed_filepath = f"{self.processed_data_path}/{symbol}_normalized.parquet"
            df_base.to_parquet(processed_filepath)
            print(f"  -> Saved normalized data for {symbol}")
            
            processed_data[symbol] = df_base
        
        print("--- Data Processing Complete ---")
        return processed_data

if __name__ == "__main__":
    loader = MultiAssetLoader()
    loader.load_and_process_data()