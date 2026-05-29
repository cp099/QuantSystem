"""
Aether Bayesian Kernel - Global Instinct Training
Public wrapper shell. If the proprietary training script is present locally, it is executed.
Otherwise, it compiles a dummy prior model for local visual dashboard testing.
"""

import sys
import os
import joblib

# Attempt to load and run the proprietary script
try:
    from scripts.train_instinct_secret import train_instinct as run_secret
except ImportError:
    run_secret = None

def train_instinct():
    if run_secret is not None:
        run_secret()
        return

    print("[KERNEL TRAINING] [IP PROTECTION] Core instinct training is closed-source.")
    print("[KERNEL TRAINING] Generating a dummy prior model for compliance and testing...")
    
    from src.brain.abmsm import ABMSM
    brain = ABMSM(K=5, D=7)
    
    os.makedirs("models", exist_ok=True)
    save_path = "models/base_instinct.pkl"
    joblib.dump(brain, save_path)
    print(f"[KERNEL TRAINING] Dummy instinct model saved to {save_path}")

if __name__ == "__main__":
    train_instinct()