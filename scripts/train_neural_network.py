"""
Aether Bayesian Kernel - Global Neural Network Training Script
Constructs and trains a global MLP model across stocks, commodities, crypto, 
and forex to establish cross-sectional directional probability mapping.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.data_loader import UniversalLoader

def train_neural_network():
    print("[NN TRAINING] INITIATING GLOBAL NEURAL NETWORK MODEL OPTIMIZATION...")
    loader = UniversalLoader()
    
    # Diverse assets to represent major global markets and asset classes
    training_tickers = [
        "SPY",      # US Large Cap Stock Index
        "QQQ",      # Tech Stock Index
        "^NSEI",    # Indian Nifty 50 Index
        "GLD",      # Gold Commodity Index
        "BTC-USD",  # Bitcoin Cryptocurrency
        "EURUSD=X"  # Major FX Currency Pair
    ]
    
    all_X = []
    all_y = []
    
    features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']
    
    for ticker in training_tickers:
        try:
            print(f"[NN TRAINING] Ingesting and engineering feature space for {ticker}...")
            df, _, _ = loader.fetch_and_engineer(ticker)
            
            if len(df) < 50:
                continue
                
            # Create forward directional label (1 if price increases next bar, else 0)
            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            # Drop the last row because it doesn't have a shift target
            df_clean = df.dropna()
            
            X = df_clean[features].values
            y = df_clean['target'].values
            
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"[NN TRAINING ERROR] Failed to process {ticker}: {e}")
            
    if not all_X:
        print("[NN TRAINING CRITICAL] No training data ingested. Aborting model compilation.")
        return
        
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    
    print(f"[NN TRAINING] Compiled dataset size: {X_train.shape[0]} samples across {X_train.shape[1]} features.")
    
    # Instantiate MLP Classifier
    # 2 Hidden layers of sizes 32 and 16 with relu activation
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    print("[NN TRAINING] Fitting Multi-Layer Perceptron neural network...")
    mlp.fit(X_train, y_train)
    
    train_score = mlp.score(X_train, y_train)
    print(f"[NN TRAINING] Model fit converged. Final accuracy: {train_score:.2%}")
    
    # Save the trained model state to persistence
    os.makedirs("models", exist_ok=True)
    save_path = "models/global_neural_network.pkl"
    joblib.dump(mlp, save_path)
    print(f"[NN TRAINING] Global Neural Network model secured at {save_path}")

if __name__ == "__main__":
    train_neural_network()
