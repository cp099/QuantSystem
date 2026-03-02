"""
Aether Bayesian Kernel - Global Instinct Training
Establishes the foundational market physics utilizing a multi-asset training set.
"""

import sys, os, yaml, joblib, pandas as pd

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM

def train_instinct():
    """
    Constructs the global baseline model (Instinct).
    
    Inhales a diverse range of market indices (Equities, Growth, Emerging) 
    to populate the Bayesian prior with relativistic market dynamics.
    """
    print("[KERNEL TRAINING] INITIATING GLOBAL RELATIVISTIC STATE TRAINING...")
    loader = UniversalLoader()
    
    # 1. Representative Data Ingestion
    master_df = []
    for t in ["SPY", "QQQ", "^NSEI"]:
        try:
            df, _, _ = loader.fetch_and_engineer(t)
            master_df.append(df)
        except: 
            continue
    
    full_data = pd.concat(master_df)
    
    # 2. Model Initialization (7-Dimensional Relativistic Space)
    brain = ABMSM(K=5, D=7)
    features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']
    
    # 3. Recursive State Propagation
    print(f"[KERNEL TRAINING] PROPAGATING {len(full_data)} MARKET PERIODS...")
    for i in range(len(full_data)):
        feats = [full_data.iloc[i][f] for f in features]
        brain.update(feats, adapt=True)
        
    # 4. Global Priority Persistence
    os.makedirs("models", exist_ok=True)
    joblib.dump(brain, "models/base_instinct.pkl")
    print("[KERNEL TRAINING] UNIVERSAL INSTINCT SECURED AT models/base_instinct.pkl")

if __name__ == "__main__":
    train_instinct()