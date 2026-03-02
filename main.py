"""
Aether Bayesian Kernel - System Orchestration Entry Point
Coordinates high-capacity historical audits across global universes, 
integrating recursive Bayesian inference with institutional risk control.
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM
from src.engine.portfolio import CapitalAllocator
from src.research.validator import StrategyValidator
from src.strategies.trend_engine import TrendEngine
from src.strategies.mean_reversion import MeanReversionEngine
from src.strategies.relative_strength import RelativeStrengthEngine
from src.strategies.volatility_breakout import VolatilityBreakoutEngine
from src.engine.universe import SECTORS

def load_config():
    """Loads institutional parameter set from persistence."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_headless_audit():
    """
    Executes a master system audit utilizing the recursive Bayesian kernel.
    
    Synthesizes multi-asset data streams and propagates them through the 
    adaptive state-space, calculating capital rebalance targets and 
    generating a final performance memorandum.
    """
    print(f"[SYSTEM KERNEL] INITIATING MASTER AUDIT: {datetime.now()}")
    config = load_config()
    
    # --- PHASE I: ARCHITECTURAL INITIALIZATION ---
    loader = UniversalLoader()
    brain = ABMSM.load(config['regime_model']['abmsm']['save_path'])
    
    # Assembly of the modular strategy suite
    engines = [
        TrendEngine(), 
        MeanReversionEngine(), 
        RelativeStrengthEngine(top_n=1), 
        VolatilityBreakoutEngine()
    ]
    controller = CapitalAllocator('config.yaml', engines)
    
    # --- PHASE II: MULTI-ASSET SYNCHRONIZATION ---
    # We select a representative sample from the Universe Map for the Master Audit
    audit_tickers = SECTORS["JURISDICTION: USA (NYSE/NASDAQ)"]["TECHNOLOGY_LEADERS"][:3] + \
                    SECTORS["JURISDICTION: INDIA (NSE)"]["CORE_EQUITIES"][:2]
    
    processed_data = {}
    for ticker in audit_tickers:
        try:
            # fetch_and_engineer is the hardened V3.5 method
            df, _, _ = loader.fetch_and_engineer(ticker)
            processed_data[ticker] = df
        except Exception as e:
            print(f"[DATA ERROR] SKIPPING {ticker}: {e}")

    if not processed_data:
        print("[CRITICAL] NO DATA LOADED. ABORTING AUDIT.")
        return

    # Utilize the primary ticker (first in list) to anchor the timeline
    primary_ticker = list(processed_data.keys())[0]
    timeline = processed_data[primary_ticker].index.sort_values()
    symbols = list(processed_data.keys())
    
    # --- PHASE III: RECURSIVE STATE PROPAGATION ---
    equity_hist = []
    initial_cap = config['portfolio']['initial_capital']
    equity = initial_cap
    cash = initial_cap
    holdings = {s: 0.0 for s in symbols}
    
    print(f"[SYSTEM KERNEL] PROPAGATING {len(timeline)} BARS ACROSS {len(symbols)} ASSETS...")
    
    # Offset to allow for rolling window features (e.g., 200-day Rel_Risk)
    for i in range(250, len(timeline)):
        date = timeline[i]
        
        # Construct current market slice safely
        market_slice = {}
        for s in symbols:
            if date in processed_data[s].index:
                market_slice[s] = processed_data[s].loc[date]
        
        if primary_ticker not in market_slice:
            continue

        # 1. LATENT STATE ESTIMATION (Adaptive Logic)
        row_p = market_slice[primary_ticker]
        feats = [row_p['v'], row_p['r'], row_p['c'], row_p['a'], row_p['d'], row_p['l'], row_p['b']]
        regime_probs = brain.update(feats, adapt=True)
        
        # 2. HIERARCHICAL CAPITAL BUDGETING
        # Utilizing the standardized Velocity (v) for correlation attribution
        try:
            hist_subset = pd.concat({s: processed_data[s].iloc[i-60:i]['v'] for s in market_slice.keys()}, axis=1)
        except:
            hist_subset = None
            
        allocs = controller.update_allocations(regime_probs, market_slice, hist_subset)
        
        # 3. CAPITAL LIQUIDATION AND POSITION REALIGNMENT
        # Mark-to-market current valuation
        current_holdings_value = 0
        for s, units in holdings.items():
            if s in market_slice:
                current_holdings_value += units * market_slice[s]['Close']
        
        equity = cash + current_holdings_value
        equity_hist.append(equity)
        
        # Target weight execution
        new_cash = equity
        for s, weight in allocs.items():
            if s in market_slice:
                target_val = equity * weight
                holdings[s] = target_val / market_slice[s]['Close']
                new_cash -= target_val
        cash = new_cash

    # --- PHASE IV: PERFORMANCE ATTRIBUTION ---
    validator = StrategyValidator()
    # Validating against the operational timeline
    metrics = validator.calculate_metrics(equity_hist, timeline[250:])
    validator.print_terminal_report("ABMSM MASTER KERNEL", metrics, metrics, "USD")

if __name__ == "__main__":
    run_headless_audit()