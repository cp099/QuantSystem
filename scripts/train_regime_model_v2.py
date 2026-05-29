"""
Aether Bayesian Kernel - Offline Anchor Training
Public wrapper shell. If the proprietary training script is present locally, it is executed.
Otherwise, it runs a baseline anchor training logic.
"""

import sys
import os

try:
    from scripts.train_regime_model_v2_secret import main as run_secret
except ImportError:
    run_secret = None

def main():
    if run_secret is not None:
        run_secret()
        return

    print("[SYSTEM ANCHOR] [IP PROTECTION] Core regime anchor training is closed-source.")
    print("[SYSTEM ANCHOR] Executing standard baseline clustering training for demo...")
    
    # Run the regime detector fit with a dummy model
    import pandas as pd
    import yaml
    from src.engine.data_loader import UniversalLoader
    from src.research.regime_engine import RegimeDetectorV2
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    loader = UniversalLoader()
    calibration_tickers = ["SPY", "QQQ"]
    all_features = []
    
    for ticker in calibration_tickers:
        try:
            df, _, _ = loader.fetch_and_engineer(ticker)
            all_features.append(df)
        except:
            continue
            
    if all_features:
        concatenated_df = pd.concat(all_features, axis=0)
        hmm_detector = RegimeDetectorV2(config)
        hmm_detector.fit(concatenated_df)
        print("[SYSTEM ANCHOR] Standard baseline clustering complete.")
    else:
        print("[SYSTEM ANCHOR] No data loaded. Aborting baseline training.")

if __name__ == "__main__":
    main()