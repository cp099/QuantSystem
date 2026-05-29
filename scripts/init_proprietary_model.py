"""
Aether Bayesian Kernel - Model Initialization Utility
Public wrapper shell. If the proprietary initialization script is present locally, it is executed.
Otherwise, it runs a baseline initialization logic.
"""

import sys
import os

try:
    from scripts.init_proprietary_model_secret import main as run_secret
except ImportError:
    run_secret = None

def main():
    if run_secret is not None:
        run_secret()
        return

    print("[KERNEL CONTROL] [IP PROTECTION] Proprietary model initialization is closed-source.")
    print("[KERNEL CONTROL] Initializing dummy model configurations...")
    
    import yaml
    import joblib
    from src.brain.abmsm import ABMSM
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    abmsm = ABMSM(
        K=config['regime_model']['n_components'],
        D=len(config['regime_model']['features']),
        alpha=config['regime_model']['abmsm']['learning_rate'],
        lam=config['regime_model']['abmsm']['decay_factor']
    )
    
    save_path = config['regime_model']['abmsm']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(abmsm, save_path)
    print(f"[KERNEL CONTROL] Dummy model saved to {save_path}")

if __name__ == "__main__":
    main()