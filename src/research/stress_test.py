import pandas as pd
import numpy as np
import os
import sys

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM
from src.engine.portfolio import PortfolioControllerV2
from src.research.validator import StrategyValidator
from src.strategies.trend_engine import TrendEngine

def run_5d_stress_test():
    print("--- INITIATING 5D BLACK SWAN STRESS TEST ---")
    
    # 1. Use the upgraded Universal Loader
    loader = UniversalLoader()
    df, bench_df, _ = loader.fetch_and_engineer("SPY")
    symbols = ["SPY"]
    
    # 2. Initialize the 5D Brain
    # Load the instinct model we trained with 5 dimensions
    brain = ABMSM.load('models/base_instinct.pkl')
    engines = [TrendEngine()] 
    controller = PortfolioControllerV2('config.yaml', engines)
    
    equity_hist = []
    bench_hist = [] 
    
    equity, bench = 100000, 100000
    cash, holdings = 100000, {"SPY": 0.0}
    
    # Audit Window
    timeline = df.index[300:700]
    start_px = df.loc[timeline[0], 'Close']

    print(f"Stress Testing across {len(timeline)} bars...")

    for i, date in enumerate(timeline):
        # Build market slice
        row = df.loc[date].copy()
        bench_row = bench_df.loc[date].copy()
        
        # --- INJECT 25% SHOCK AT BAR 150 ---
        is_shocked = False
        if 150 <= i <= 160:
            is_shocked = True
            row['Close'] *= 0.75 
            row['Velocity'] = -5.0 # High speed crash
            row['Rel_Risk'] *= 4.0 # Massive vol spike
            row['Vol_Div'] *= 3.0  # Divergence sense

        # 1. Benchmark (Buy & Hold)
        bench = (row['Close'] / start_px) * 100000
        bench_hist.append(bench)

        # 2. 5-Dimensional Brain Update
        # Senses: [Velocity, Rel_Risk, Compression, Rel_Alpha, Vol_Div]
        feats = [row['Velocity'], row['Rel_Risk'], row['Compression'], row['Rel_Alpha'], row['Vol_Div']]
        probs = brain.update(feats)
        
        # 3. Alpha Veto Logic
        bull_states = brain.get_bull_states()
        p_growth = sum(probs[s] for s in bull_states)
        entropy = brain.get_entropy()
        
        # Veto trade if underperforming or confused
        signal = p_growth
        if row['Rel_Alpha'] < -0.005 or entropy > 0.88:
            signal = 0.0

        # 4. Adaptive Portfolio Update
        current_val = cash + (holdings['SPY'] * row['Close'])
        equity_hist.append(current_val)
        
        # Rebalance
        target_shares = (current_val * signal) / row['Close']
        diff = target_shares - holdings['SPY']
        cash -= diff * row['Close']
        holdings['SPY'] = target_shares

        if is_shocked and i == 150:
            print(f"[SHOCK INJECTED] Date: {date.date()} | Brain Entropy: {entropy:.4f}")

    # Final Comparative Report
    v = StrategyValidator()
    m_sys = v.calculate_metrics(equity_hist)
    m_bnch = v.calculate_metrics(bench_hist)
    
    print(f"\n{'='*55}")
    print(f"{'METRIC':<20} | {'5D ADAPTIVE':<15} | {'BENCHMARK':<15}")
    print(f"{'-'*55}")
    print(f"{'Final Equity':<20} | ${m_sys['Final']:,.0f} | ${m_bnch['Final']:,.0f}")
    print(f"{'Max Drawdown':<20} | {m_sys['MaxDD']:<15} | {m_bnch['MaxDD']:<15}")
    print(f"{'Total Return':<20} | {m_sys['Return']:<15} | {m_bnch['Return']:<15}")
    print(f"{'='*55}")
    
    print("\n>>> STATUS: 5-SENSE VALIDATION COMPLETE.")

if __name__ == "__main__":
    run_5d_stress_test()