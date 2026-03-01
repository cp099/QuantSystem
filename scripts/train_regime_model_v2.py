import pandas as pd
import yaml
import sys
import os

# Fix path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.data_loader import MultiAssetLoader
from src.research.regime_engine import RegimeDetectorV2

def main():
    print("--- RE-MINTING REGIME MODEL V2 ---")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    loader = MultiAssetLoader('config.yaml')
    processed_data_dict = loader.load_and_process_data()
    
    all_features = []
    for symbol, df in processed_data_dict.items():
        df_with_symbol = df.copy()
        df_with_symbol['symbol'] = symbol
        all_features.append(df_with_symbol)
        
    concatenated_df = pd.concat(all_features, axis=0)
    
    # This now uses src.research.regime_engine
    hmm_detector = RegimeDetectorV2(config)
    hmm_detector.fit(concatenated_df)
    print("\n--- REGIME MODEL V2 SAVED ---")

if __name__ == "__main__":
    main()