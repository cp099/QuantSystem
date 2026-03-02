"""
Aether Bayesian Kernel - Model Initialization Utility
Synchronizes the 7-dimensional offline priors with the adaptive proprietary kernel.
"""

import sys
import os
import yaml
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.brain.abmsm import ABMSM
from src.research.regime_engine import RegimeDetectorV2

def main():
    print("[KERNEL CONTROL] INITIALIZING 7-SENSE PROPRIETARY KERNEL...")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    offline_detector = RegimeDetectorV2(config)
    offline_detector.load()

    # Kernel Instantiation (D=7)
    abmsm = ABMSM(
        K=config['regime_model']['n_components'],
        D=len(config['regime_model']['features']),
        alpha=config['regime_model']['abmsm']['learning_rate'],
        lam=config['regime_model']['abmsm']['decay_factor']
    )
    
    # Explicit 7-Dimensional Knowledge Transfer
    abmsm.means = offline_detector.model.means_
    abmsm.covs = offline_detector.model.covars_
    abmsm.A = offline_detector.model.transmat_
    
    save_path = config['regime_model']['abmsm']['save_path']
    joblib.dump(abmsm, save_path)
    print(f"[KERNEL CONTROL] 7-SENSE KERNEL PERSISTED: {save_path}")

if __name__ == "__main__":
    main()