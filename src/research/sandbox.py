"""
Aether Bayesian Kernel - Strategy Audit Pipeline
Coordinates a high-fidelity historical simulation for a specific security identifier, 
synthesizing Bayesian state estimation with institutional risk control.
"""

import sys, os, argparse, yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM
from src.research.validator import StrategyValidator
from src.engine.risk_manager import RiskManager
from src.research.validator_pro import AdvancedValidator
from src.research.sentinel_report import SentinelReport

def run_universal_sandbox(ticker):
    """
    Executes a comprehensive performance audit for a specific asset.
    
    Coordinates the recursive processing of relativistic features through the 
    Bayesian kernel and calculates risk-adjusted capital exposure based on 
    volatility targeting and drawdown feedback.

    Args:
        ticker (str): Global security identifier.
    """
    # --- PHASE I: ARCHITECTURAL INITIALIZATION ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = UniversalLoader()
    df, bench_df, local_ccy = loader.fetch_and_engineer(ticker)
    
    try:
        brain = ABMSM.load(config['regime_model']['instinct_save_path'])
    except FileNotFoundError:
        print("[AUDIT KERNEL] CRITICAL: BASE INSTINCT MODEL NOT DETECTED.")
        return

    # Risk kernel utilizes a 12% target volatility mandate
    risk_engine = RiskManager(target_vol=0.12)
    
    # Currency-basis initialization
    initial_cap_local = 100000.0
    initial_fx = df['FX_Rate'].iloc[0]
    initial_cap_usd = initial_cap_local / initial_fx
    
    equity_local, cash_local, shares = initial_cap_local, initial_cap_local, 0.0
    hist_local, hist_usd, bench_hist = [], [], []
    
    features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']
    peak_equity = initial_cap_local
    
    print(f"[AUDIT KERNEL] INITIATING STRATEGY AUDIT: {ticker} (BASE USD: ${initial_cap_usd:,.2f})")
    
    # --- PHASE II: RECURSIVE SIMULATION CYCLE ---
    for i in range(len(df)):
        row = df.iloc[i]
        feats = [row[f] for f in features]
        
        # Recursive Kernel Update
        brain.update(feats, adapt=True)
        
        # Directional Conviction Generation
        signal = brain.get_bayesian_signal()
        
        # Capital Liquidation Calculation
        current_val_local = cash_local + (shares * row['Close'])
        
        # Drawdown Feedback Calculation
        if current_val_local > peak_equity:
            peak_equity = current_val_local
        current_dd = (current_val_local - peak_equity) / peak_equity
        
        # State-Space Calibration Buffer (120-day observation window)
        if i < 120: 
            signal = 0.0
        
        # Adaptive Capital Allocation
        target_value = risk_engine.calculate_position_size(
            current_val_local, 
            row['vol_pct'], 
            signal, 
            current_dd
        )
        
        # Rebalance Execution
        target_shares = target_value / row['Close']
        shares_diff = target_shares - shares
        cash_local -= (shares_diff * row['Close'])
        shares = target_shares
            
        # Temporal State Persistence
        hist_local.append(current_val_local)
        hist_usd.append(current_val_local / row['FX_Rate'])
        bench_hist.append((bench_df.iloc[i]['Close'] / bench_df['Close'].iloc[0]) * initial_cap_local)

    # --- PHASE III: PERFORMANCE ANALYTICS & REPORTING ---
    validator = StrategyValidator()
    m_local = validator.calculate_metrics(hist_local, df.index)
    m_usd = validator.calculate_metrics(hist_usd, df.index)
    
    # Cross-currency return normalization
    usd_final_val = hist_usd[-1]
    usd_ret_pct = (usd_final_val / initial_cap_usd - 1) * 100
    m_usd['Return'] = f"{usd_ret_pct:.2f}%"

    # Terminal Audit Presentation
    validator.print_terminal_report(ticker, m_local, m_usd, local_ccy)
    
    # Statistical Path Stress-Testing (Monte Carlo)
    pro_metrics = AdvancedValidator.run_monte_carlo(hist_local)
    print("\n--- STATISTICAL PATH ROBUSTNESS (1,000 SIMULATIONS) ---")
    for k, v in pro_metrics.items():
        print(f"{k:<20}: {v}")
    
    # Institutional Document Compilation
    reporter = SentinelReport(ticker, local_ccy)
    reporter.build_report(hist_local, bench_hist, m_local, m_usd)
    print(f"[AUDIT KERNEL] INSTITUTIONAL MEMORANDUM EXPORTED TO /REPORTS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()
    run_universal_sandbox(args.ticker)