import sys
import os
import yaml
import joblib
import pandas as pd

# Fix path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM

def train_instinct():
    print(">>> TRAINING GLOBAL RELATIVISTIC INSTINCT MODEL ON S&P 500...")
    loader = UniversalLoader()
    
    # Unpack the 3 values returned by the upgraded loader
    df, _, _ = loader.fetch_and_engineer("SPY")
    
    # Initialize Brain with 3 Relativistic Senses
    # D=3 corresponds to [Velocity, Rel_Risk, Compression]
    brain = ABMSM(K=5, D=5)
    
    print(f"Feeding {len(df)} days of market physics into Brain...")
    for i in range(len(df)):
        row = df.iloc[i]
        
        # --- FIX: Use the NEW standardized keys ---
        # Velocity: Directional momentum normalized by ATR
        # Rel_Risk: Current volatility relative to historical mean
        # Compression: Squeeze factor (Vol Ratio)
        feats = [row['Velocity'], row['Rel_Risk'], row['Compression'], row['Rel_Alpha'], row['Vol_Div']]
        
        brain.update(feats)
        
    os.makedirs("models", exist_ok=True)
    joblib.dump(brain, "models/base_instinct.pkl")
    print(">>> RELATIVISTIC INSTINCT SAVED TO models/base_instinct.pkl")

if __name__ == "__main__":
    train_instinct()