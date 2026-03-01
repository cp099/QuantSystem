import sys
import os
import yaml
import joblib

# Fix path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.brain.abmsm import ABMSM
from src.research.regime_engine import RegimeDetectorV2

def main():
    print("--- RE-MINTING PROPRIETARY BRAIN (ABMSM) ---")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    offline_detector = RegimeDetectorV2(config)
    offline_detector.load()

    # This now uses src.brain.abmsm
    abmsm = ABMSM(
        K=config['regime_model']['n_components'],
        D=len(config['regime_model']['features']),
        alpha=config['regime_model']['abmsm']['learning_rate'],
        lam=config['regime_model']['abmsm']['decay_factor']
    )
    
    # Warm start from HMM
    abmsm.means = offline_detector.model.means_
    abmsm.covs = offline_detector.model.covars_
    abmsm.A = offline_detector.model.transmat_
    
    save_path = config['regime_model']['abmsm']['save_path']
    joblib.dump(abmsm, save_path)
    print(f"Proprietary model re-minted at {save_path}")

if __name__ == "__main__":
    main()