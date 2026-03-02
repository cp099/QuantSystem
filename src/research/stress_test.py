"""
Aether Bayesian Kernel - Black Swan Stress Engine
Implements synthetic tail-risk injection to validate the recursive 
adaptation of the Bayesian kernel during extreme market dislocation.
"""

import pandas as pd
import numpy as np
import os
import sys

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM
from src.engine.portfolio import CapitalAllocator
from src.research.validator import StrategyValidator
from src.strategies.trend_engine import TrendEngine

def run_7d_stress_test():
    """
    Executes an institutional stress test via synthetic shock injection.
    
    Inhales SPY historical data and injects a 25% price collapse and 
    volatility explosion to monitor the latency and accuracy of the 
    kernel's defensive transition.
    """
    print("[STRESS TEST] INITIATING 7-SENSE TAIL-RISK VALIDATION...")
    
    # --- PHASE I: COMPONENT INITIALIZATION ---
    loader = UniversalLoader()
    df, bench_df, _ = loader.fetch_and_engineer("SPY")
    
    # Restoring pre-trained global instinct (7-Dimensional)
    brain = ABMSM.load('models/base_instinct.pkl')
    engines = [TrendEngine()] 
    controller = CapitalAllocator('config.yaml', engines)
    
    equity_hist, bench_hist = [], []
    equity, bench = 100000, 100000
    cash, holdings = 100000, {"SPY": 0.0}
    
    # Defining specific audit window for shock observation
    timeline = df.index[300:700]
    start_px = df.loc[timeline[0], 'Close']

    print(f"[STRESS TEST] PROPAGATING {len(timeline)} BARS UNDER ADVERSARIAL CONDITIONS...")

    # --- PHASE II: ADVERSARIAL PROPAGATION ---
    for i, date in enumerate(timeline):
        row = df.loc[date].copy()
        
        # SYNTHETIC TAIL-RISK INJECTION
        # Simulates a -25% gap down and correlated systemic stress
        is_shocked = False
        if 150 <= i <= 155:
            is_shocked = True
            row['Close'] *= 0.75 
            row['v'] = -4.0 # Maximum negative velocity clamp
            row['r'] = 4.0  # Maximum risk spike clamp
            row['b'] = -4.0 # Participation collapse
            row['l'] = -4.0 # Liquidity withdrawal

        # Market Benchmark Update (Linear Buy & Hold)
        bench = (row['Close'] / start_px) * 100000
        bench_hist.append(bench)

        # Bayesian State Update (7 Senses)
        # Vector: [Velocity, Risk, Compression, Alpha, VolDiv, Liquidity, Breadth]
        feats = [row['v'], row['r'], row['c'], row['a'], row['d'], row['l'], row['b']]
        probs = brain.update(feats)
        
        # Proprietary Signal Generation
        signal = brain.get_bayesian_signal()
        entropy = brain.get_entropy()
        
        # Institutional Alpha Veto
        if row['a'] < -0.01: 
            signal = 0.0

        # Capital Liquidation Calculation
        current_val = cash + (holdings['SPY'] * row['Close'])
        equity_hist.append(current_val)
        
        # Position Realignment
        target_shares = (current_val * signal) / row['Close']
        diff = target_shares - holdings['SPY']
        cash -= diff * row['Close']
        holdings['SPY'] = target_shares

        if is_shocked and i == 150:
            print(f"[SHOCK ALERT] T+0 SHOCK DETECTED. BRAIN ENTROPY: {entropy:.4f}")

    # --- PHASE III: COMPARATIVE AUDIT ---
    v = StrategyValidator()
    m_sys = v.calculate_metrics(equity_hist, timeline)
    m_bnch = v.calculate_metrics(bench_hist, timeline)
    
    width = 65
    print("\n" + "="*width)
    print(f" STRESS TEST COMPARISON: 7-SENSE ADAPTIVE VS BENCHMARK")
    print("-" * width)
    print(f"{'METRIC':<20} | {'7D ADAPTIVE':<18} | {'BENCHMARK':<15}")
    print("-" * width)
    
    results = [
        ("Final Equity", f"${m_sys['Final']:,.0f}", f"${m_bnch['Final']:,.0f}"),
        ("Max Drawdown", m_sys['MaxDD'], m_bnch['MaxDD']),
        ("Total Return", m_sys['Return'], m_bnch['Return'])
    ]
    
    for name, s, b in results:
        print(f"{name:<20} | {s:<18} | {b:<15}")
    
    print("="*width)
    print("\n[STRESS TEST] RESILIENCE VALIDATION COMPLETE.")

if __name__ == "__main__":
    run_7d_stress_test()