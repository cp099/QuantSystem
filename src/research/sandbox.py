import sys, os, argparse, yaml
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM
from src.research.validator import StrategyValidator

def run_universal_sandbox(ticker):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = UniversalLoader()
    df, bench_df, local_ccy = loader.fetch_and_engineer(ticker)
    brain = ABMSM.load(config['regime_model']['instinct_save_path'])
    
    # --- PROPER CURRENCY INITIALIZATION ---
    initial_cap_local = 100000
    initial_fx = df['FX_Rate'].iloc[0]
    initial_cap_usd = initial_cap_local / initial_fx
    
    equity_local, cash_local, shares = initial_cap_local, initial_cap_local, 0
    hist_local, hist_usd = [], []
    
    print(f">>> Audit Started. Base USD Capital: ${initial_cap_usd:,.2f}")
    
    for i in range(len(df)):
        row = df.iloc[i]
        # 5 Senses: Velocity, Risk, Compression, Alpha, Vol_Div
        feats = [row['Velocity'], row['Rel_Risk'], row['Compression'], row['Rel_Alpha'], row['Vol_Div']]
        
        probs = brain.update(feats, adapt=True)
        bull_states = brain.get_bull_states()
        p_growth = sum(probs[s] for s in bull_states)
        entropy = brain.get_entropy()
        
        # --- REFINED INSTITUTIONAL LOGIC ---
        # 1. Base Signal
        signal = p_growth
        # 2. Hard Alpha Veto
        if row['Rel_Alpha'] < -0.005: signal = 0.0 
        # 3. Uncertainty Veto
        if entropy > 0.88: signal = 0.0
        
        if i < 252: signal = 0.0 # Warmup

        # Execution
        current_val_local = cash_local + (shares * row['Close'])
        target_shares = (current_val_local * signal) / row['Close']
        shares_diff = target_shares - shares
        cash_local -= shares_diff * row['Close']
        shares = target_shares
            
        hist_local.append(current_val_local)
        # Proper USD Value
        hist_usd.append(current_val_local / row['FX_Rate'])

    validator = StrategyValidator()
    m_local = validator.calculate_metrics(hist_local)
    m_usd = validator.calculate_metrics(hist_usd)
    
    # Calculate true USD Return relative to USD start
    usd_return = (hist_usd[-1] / initial_cap_usd - 1) * 100
    m_usd['Return'] = f"{usd_return:.2f}%"
    
    validator.print_dual_report(ticker, m_local, m_usd, local_ccy)
    from src.research.sentinel_report import SentinelReport
    reporter = SentinelReport(ticker)
    reporter.generate(hist_local, hist_usd, bench_hist, local_ccy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()
    run_universal_sandbox(args.ticker)