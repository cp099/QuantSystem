import sys, os, argparse, yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Fix path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM
from src.research.validator import StrategyValidator
from src.engine.risk_manager import RiskManager
from src.research.validator_pro import AdvancedValidator
from src.research.sentinel_report import SentinelReport

def run_universal_sandbox(ticker):
    # 1. SETUP & CONFIG
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = UniversalLoader()
    df, bench_df, local_ccy = loader.fetch_and_engineer(ticker)
    
    # Load Instinct (S&P 500 Knowledge)
    try:
        brain = ABMSM.load(config['regime_model']['instinct_save_path'])
    except FileNotFoundError:
        print("CRITICAL: Instinct model not found. Run scripts/train_instinct.py first.")
        return

    # Institutional Risk Engine (Targeting 12% Vol for Iron Clad stability)
    risk_engine = RiskManager(target_vol=0.12)
    
    # Initial Capital Calculations
    initial_cap_local = 100000.0
    initial_fx = df['FX_Rate'].iloc[0]
    initial_cap_usd = initial_cap_local / initial_fx
    
    equity_local, cash_local, shares = initial_cap_local, initial_cap_local, 0.0
    hist_local, hist_usd, bench_hist = [], [], []
    
    features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']
    peak_equity = initial_cap_local
    
    print(f">>> SOVEREIGN AUDIT: {ticker} | Start USD: ${initial_cap_usd:,.2f}")
    
    # 2. SIMULATION LOOP
    for i in range(len(df)):
        row = df.iloc[i]
        feats = [row[f] for f in features]
        
        # A. Brain Update (Adaptive Learning)
        brain.update(feats, adapt=True)
        
        # B. Get Bayesian convictions
        signal = brain.get_bayesian_signal()
        
        # C. Calculate Current State (MTM)
        current_val_local = cash_local + (shares * row['Close'])
        
        # D. Drawdown Calculation (Relative to highest equity seen)
        if current_val_local > peak_equity:
            peak_equity = current_val_local
        current_dd = (current_val_local - peak_equity) / peak_equity
        
        # E. Warmup & Logic Veto
        # No trading for first 120 days to allow Z-scores and Brain to calibrate
        if i < 120: 
            signal = 0.0
        
        # F. Risk-Adjusted Sizing (Phase 13: Vol-Target + Drawdown Penalty)
        target_value = risk_engine.calculate_position_size(
            current_val_local, 
            row['vol_pct'], 
            signal, 
            current_dd
        )
        
        # G. Rebalance Execution
        target_shares = target_value / row['Close']
        shares_diff = target_shares - shares
        cash_local -= (shares_diff * row['Close'])
        shares = target_shares
            
        # H. History Tracking
        hist_local.append(current_val_local)
        hist_usd.append(current_val_local / row['FX_Rate'])
        
        # Benchmark comparison (Relative to 100k)
        bench_hist.append((bench_df.iloc[i]['Close'] / bench_df['Close'].iloc[0]) * initial_cap_local)

    # 3. VALIDATION & REPORTING
    validator = StrategyValidator()
    m_local = validator.calculate_metrics(hist_local, df.index)
    m_usd = validator.calculate_metrics(hist_usd, df.index)
    
    # Accurate USD Return relative to original USD principal
    usd_final_val = hist_usd[-1]
    usd_ret_pct = (usd_final_val / initial_cap_usd - 1) * 100
    m_usd['Return'] = f"{usd_ret_pct:.2f}%"

    # Terminal Output
    validator.print_terminal_report(ticker, m_local, m_usd, local_ccy)
    
    # Advanced Monte Carlo Proof
    pro_metrics = AdvancedValidator.run_monte_carlo(hist_local)
    print(">>> MONTE CARLO STRESS TEST (1,000 PATHS)")
    for k, v in pro_metrics.items():
        print(f"{k:<20}: {v}")
    print("-" * 65)

    # Generate Institutional PDF Dossier
    reporter = SentinelReport(ticker, local_ccy)
    reporter.build_report(hist_local, bench_hist, m_local, m_usd)
    print(f">>> Audit Exported to /reports")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()
    run_universal_sandbox(args.ticker)