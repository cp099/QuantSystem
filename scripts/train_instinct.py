import sys, os, yaml, joblib, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM

def train_instinct():
    print(">>> TRAINING GLOBAL 7-SENSE KERNEL (SPY + QQQ + NIFTY)...")
    loader = UniversalLoader()
    
    master_df = []
    for t in ["SPY", "QQQ", "^NSEI"]:
        try:
            df, _, _ = loader.fetch_and_engineer(t)
            master_df.append(df)
        except: continue
    
    full_data = pd.concat(master_df)
    brain = ABMSM(K=5, D=7)
    features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']
    
    print(f"Propagating {len(full_data)} market days...")
    for i in range(len(full_data)):
        feats = [full_data.iloc[i][f] for f in features]
        brain.update(feats, adapt=True)
        
    os.makedirs("models", exist_ok=True)
    joblib.dump(brain, "models/base_instinct.pkl")
    print(">>> UNIVERSAL SOVEREIGN KERNEL SAVED.")

if __name__ == "__main__":
    train_instinct()