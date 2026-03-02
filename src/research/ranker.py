import sys, os, yaml, pandas as pd, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM

class SectorRanker:
    def __init__(self):
        with open('config.yaml', 'r') as f: self.config = yaml.safe_load(f)
        self.loader = UniversalLoader()

    def rank_universe(self, tickers):
        rankings = []
        features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']
        for ticker in tickers:
            try:
                brain = ABMSM.load(self.config['regime_model']['instinct_save_path'])
                df, _, _ = self.loader.fetch_and_engineer(ticker)
                
                # Calibration (Institutional Standard: 120 Days)
                calib = df.tail(121)
                for i in range(len(calib)-1):
                    brain.update([calib.iloc[i][f] for f in features])
                
                probs = brain.update([calib.iloc[-1][f] for f in features])
                bull_idx = brain.get_bull_states()
                p_bull = sum(probs[i] for i in bull_idx)
                entropy = brain.get_entropy()
                
                # 20-Day Momentum Gate
                raw_20d_ret = df['Close'].pct_change(20).iloc[-1]
                
                # --- INSTITUTIONAL ALPHA VETO ---
                # Score is 0 if either the Brain is Bearish OR the Price is dropping
                if p_bull < 0.40 or raw_20d_ret < 0:
                    score = 0.0
                else:
                    score = p_bull * (1.0 - entropy)
                
                rankings.append({
                    "Ticker": ticker, "Score": round(score, 3),
                    "Bull_P": f"{p_bull:.1%}", "Uncert": round(entropy, 2),
                    "20D_Ret": f"{raw_20d_ret:.2%}"
                })
            except: continue
        return pd.DataFrame(rankings).sort_values(by="Score", ascending=False)