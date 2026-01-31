import pandas as pd
import yaml
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import MultiAssetLoader
from src.regime_v2 import RegimeDetectorV2

def main():
    print("--- Starting V2 Regime Model Training Pipeline ---")
    
    # 1. Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Load all processed (normalized) data
    loader = MultiAssetLoader(config_path='config.yaml')
    processed_data_dict = loader.load_and_process_data()
    
    # 3. Concatenate features from all assets into one DataFrame
    all_features = []
    for symbol, df in processed_data_dict.items():
        # Add a symbol column to track sequence lengths later
        df_with_symbol = df.copy()
        df_with_symbol['symbol'] = symbol
        all_features.append(df_with_symbol)
        
    concatenated_df = pd.concat(all_features, axis=0)
    
    # 4. Initialize and train the HMM
    hmm_detector = RegimeDetectorV2(config)
    hmm_detector.fit(concatenated_df)
    
    print("\n--- Training Pipeline Complete ---")

if __name__ == "__main__":
    main()