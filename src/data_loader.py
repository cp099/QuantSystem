import yfinance as yf
import pandas as pd
import os

# Define storage path
RAW_DATA_PATH = "data/raw"

def ensure_directories():
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)

def download_ticker_data(tickers, start_date="2015-01-01", end_date="2024-01-01"):
    """
    Downloads data for a list of tickers and saves them as Parquet files.
    """
    ensure_directories()
    
    print(f"Starting download for {len(tickers)} tickers...")
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        try:
            # Auto_adjust=True handles splits and dividends automatically
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if df.empty:
                print(f"Warning: No data for {ticker}")
                continue
            
            # Save to Parquet
            # We use the ticker as the filename
            file_path = f"{RAW_DATA_PATH}/{ticker}.parquet"
            df.to_parquet(file_path)
            print(f"-> Saved to {file_path}")
            
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

if __name__ == "__main__":
    # Test Universe:
    # SPY (Market), TLT (Bonds), GLD (Gold), VNQ (Real Estate), USO (Oil)
    # This gives us a multi-asset view for future regime detection.
    universe = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ', 'USO']
    
    download_ticker_data(universe)