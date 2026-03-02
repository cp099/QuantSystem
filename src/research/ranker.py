"""
Aether Bayesian Kernel - Cross-Sectional Alpha Ranker
Coordinates multi-asset alpha discovery by performing sequential Bayesian 
calibration and relative performance scoring across defined universes.
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM

class SectorRanker:
    """
    Performs comparative analysis of assets using adaptive Bayesian inference.
    
    Inhales a collection of tickers, synchronizes them with the global 
    instinct model, and ranks them based on latent state belief and 
    realized momentum.
    """

    def __init__(self):
        """
        Initializes the ranking kernel and data infrastructure.
        """
        with open('config.yaml', 'r') as f: 
            self.config = yaml.safe_load(f)
        self.loader = UniversalLoader()

    def rank_universe(self, tickers):
        """
        Executes the ranking pipeline for a specific asset collection.
        
        Args:
            tickers (list): List of global security identifiers.
            
        Returns:
            pd.DataFrame: Sorted rankings containing scores and diagnostic metrics.
        """
        rankings = []
        features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']
        
        for ticker in tickers:
            try:
                # --- PHASE I: INFERENCE SEED LOADING ---
                # Restores the global prior for specific asset calibration
                brain = ABMSM.load(self.config['regime_model']['instinct_save_path'])
                df, _, _ = self.loader.fetch_and_engineer(ticker)
                
                # --- PHASE II: SEQUENTIAL CALIBRATION ---
                # Executes a 120-day observation window to adapt kernel 
                # parameters to asset-specific signatures.
                calib = df.tail(121)
                for i in range(len(calib)-1):
                    brain.update([calib.iloc[i][f] for f in features])
                
                # --- PHASE III: CROSS-SECTIONAL SCORING ---
                probs = brain.update([calib.iloc[-1][f] for f in features])
                bull_idx = brain.get_bull_states()
                p_bull = sum(probs[i] for i in bull_idx)
                entropy = brain.get_entropy()
                
                # Realized 20-period absolute momentum
                raw_20d_ret = df['Close'].pct_change(20).iloc[-1]
                
                # INFERENCE GATING: Disqualifies assets with bearish state 
                # dominance or negative realized momentum.
                if p_bull < 0.40 or raw_20d_ret < 0:
                    score = 0.0
                else:
                    score = p_bull * (1.0 - entropy)
                
                rankings.append({
                    "Ticker": ticker, 
                    "Score": round(score, 3),
                    "State_Belief": f"{p_bull:.1%}", 
                    "Entropy": round(entropy, 2),
                    "20D_Ret": f"{raw_20d_ret:.2%}"
                })
            except: 
                continue
                
        return pd.DataFrame(rankings).sort_values(by="Score", ascending=False)