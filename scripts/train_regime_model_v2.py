"""
Aether Bayesian Kernel - Offline Anchor Training
Optimizes the foundational Hidden Markov Model (HMM) utilizing the 7-Sense state-space.
"""

import pandas as pd
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.data_loader import UniversalLoader
from src.research.regime_engine import RegimeDetectorV2

def main():
    print("[SYSTEM ANCHOR] INITIATING 7-DIMENSIONAL OFFLINE OPTIMIZATION...")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    loader = UniversalLoader()
    
    # --- FIX: Explicitly loading cross-sectional data for calibration ---
    calibration_tickers = ["SPY", "QQQ", "^NSEI", "GLD"]
    all_features = []
    
    for ticker in calibration_tickers:
        try:
            df, _, _ = loader.fetch_and_engineer(ticker)
            df['symbol'] = ticker
            all_features.append(df)
        except Exception as e:
            print(f"[DATA ERROR] Failed to calibrate {ticker}: {e}")
        
    concatenated_df = pd.concat(all_features, axis=0)
    
    # Anchor HMM optimization
    hmm_detector = RegimeDetectorV2(config)
    hmm_detector.fit(concatenated_df)
    
    print("[SYSTEM ANCHOR] 7-SENSE OFFLINE OPTIMIZATION COMPLETE.")

if __name__ == "__main__":
    main()