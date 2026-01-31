import sys
import os
import yaml
import joblib

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.proprietary.abmsm import ABMSM
from src.regime_v2 import RegimeDetectorV2

def main():
    print("--- Initializing Proprietary ABMSM Core ---")
    
    # 1. Load Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Load the Offline HMM (V2)
    # We use the class wrapper to find the path, but we load the raw sklearn model
    offline_detector = RegimeDetectorV2(config)
    try:
        offline_detector.load()
    except FileNotFoundError:
        print("CRITICAL: Offline HMM model not found. Run scripts/train_regime_model_v2.py first.")
        return

    # 3. Instantiate the Proprietary ABMSM
    abmsm = ABMSM(
        n_regimes=config['regime_model']['n_components'],
        n_features=len(config['regime_model']['features']),
        learning_rate=config['regime_model']['abmsm']['learning_rate'],
        decay_factor=config['regime_model']['abmsm']['decay_factor']
    )
    
    # 4. Inject Knowledge (The Bridge)
    # We pass the internal hmmlearn model object to the ABMSM
    abmsm.initialize_from_hmm(offline_detector.model)
    
    # 5. Save the Initialized Brain
    save_path = config['regime_model']['abmsm']['save_path']
    abmsm.save(save_path)
    print(f"Proprietary model initialized and encrypted/saved to {save_path}")

if __name__ == "__main__":
    main()