import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Transforms raw OHLCV data into statistical features for Regime Detection.
    Handles MultiIndex flattening automatically.
    """
    
    def __init__(self, data):
        self.df = data.copy()
        
        # --- FIX: FLATTEN MULTI-INDEX COLUMNS ---
        if isinstance(self.df.columns, pd.MultiIndex):
            # If columns are ('Close', 'SPY'), take just 'Close'
            self.df.columns = self.df.columns.get_level_values(0)
            
        # Ensure column names are Title Case for consistency
        self.df.columns = [c.capitalize() for c in self.df.columns]

    def add_volatility_features(self, short_window=20, long_window=60):
        # Log Returns
        self.df['Log_ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # Realized Volatility
        self.df['Vol_short'] = self.df['Log_ret'].rolling(window=short_window).std() * np.sqrt(252)
        self.df['Vol_long'] = self.df['Log_ret'].rolling(window=long_window).std() * np.sqrt(252)
        
        # Volatility Ratio
        self.df['Vol_ratio'] = self.df['Vol_short'] / self.df['Vol_long']
        return self

    def add_trend_features(self, window=50):
        # Momentum
        self.df['Momentum'] = self.df['Close'].pct_change(periods=window)
        
        # Distance from MA
        ma = self.df['Close'].rolling(window=window).mean()
        self.df['Dist_ma'] = (self.df['Close'] - ma) / ma
        return self

    def add_volume_features(self, window=20):
        # Relative Volume
        # Handle case where Volume might be missing
        if 'Volume' in self.df.columns:
            vol_ma = self.df['Volume'].rolling(window=window).mean()
            self.df['Rvol'] = self.df['Volume'] / vol_ma
        else:
            self.df['Rvol'] = 1.0 # Default if no volume
        return self

    def get_features(self):
        # Return only the calculated features
        cols = ['Log_ret', 'Vol_short', 'Vol_long', 'Vol_ratio', 'Momentum', 'Dist_ma', 'Rvol']
        # Filter for cols that actually exist
        cols = [c for c in cols if c in self.df.columns]
        return self.df[cols].dropna()

# Test block
if __name__ == "__main__":
    df = pd.read_parquet("data/raw/SPY.parquet")
    fe = FeatureEngineer(df)
    fe.add_volatility_features().add_trend_features().add_volume_features()
    print(fe.get_features().tail())