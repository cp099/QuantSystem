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
from src.strategies.deep_learning import DeepLearningEngine
from src.engine.universe import SECTORS
from src.engine.compliance import PreTradeGateway
from src.engine.sentinel import Sentinel

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
        VolatilityBreakoutEngine(),
        DeepLearningEngine()
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
    
    compliance_gate = PreTradeGateway(max_weight=0.30, min_equity_floor=70000.0)
    # Clear previous audit logs for fresh dashboard visualization
    audit_file = "logs/audit/decision_audit.jsonl"
    if os.path.exists(audit_file):
        try:
            os.remove(audit_file)
        except Exception:
            pass
    sentinel = Sentinel()
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
        
        # 2. HIERARCHICAL CAPITAL BUDGETING (Weekly Rebalancing to prevent transaction cost drag)
        if i == 250 or i % 5 == 0:
            try:
                hist_subset = pd.concat({s: processed_data[s].iloc[i-60:i]['v'] for s in market_slice.keys()}, axis=1)
            except:
                hist_subset = None
                
            allocs = controller.update_allocations(regime_probs, market_slice, hist_subset)
        else:
            # Maintain current holdings weights to prevent daily churning
            allocs = {s: (holdings[s] * market_slice[s]['Close']) / equity if equity > 0 else 0.0 for s in symbols}
        
        # 3. CAPITAL LIQUIDATION AND POSITION REALIGNMENT
        # Mark-to-market current valuation
        current_holdings_value = 0
        for s, units in holdings.items():
            if s in market_slice:
                current_holdings_value += units * market_slice[s]['Close']
        
        equity = cash + current_holdings_value
        
        # Enforce Pre-Trade Compliance constraints (SEC Rule 15c3-5 & SEBI caps)
        validated_allocs = compliance_gate.validate_allocations(allocs, equity)
        
        # Target weight execution with slippage and transaction fees
        new_cash = cash
        new_holdings = holdings.copy()
        for s, weight in validated_allocs.items():
            if s in market_slice:
                current_price = market_slice[s]['Close']
                current_val = holdings[s] * current_price
                target_val = equity * weight
                trade_val = target_val - current_val
                
                # Only execute trades that change the asset allocation by >2% of equity (noise filter)
                if abs(trade_val) > 0.02 * equity:
                    compliance_gate.total_orders += 1
                    if not compliance_gate.check_otr():
                        continue
                    
                    compliance_gate.total_trades += 1
                    asset_df = processed_data[s]
                    pos = asset_df.index.get_loc(date)
                    prev_close = asset_df['Close'].iloc[pos-1] if pos > 0 else current_price
                    if not compliance_gate.validate_price_band(s, current_price, prev_close):
                        continue
                    
                    vol = market_slice[s]['vol_pct']
                    slippage = abs(trade_val) * (0.05 * vol)
                    brokerage = abs(trade_val) * 0.0005
                    total_friction = slippage + brokerage
                    
                    new_cash -= (trade_val + total_friction)
                    new_holdings[s] = target_val / current_price
        
        cash = new_cash
        holdings = new_holdings
        
        # Recalculate equity post-trade
        current_holdings_value = 0
        for s, units in holdings.items():
            if s in market_slice:
                current_holdings_value += units * market_slice[s]['Close']
        equity = cash + current_holdings_value
        equity_hist.append(equity)
        sentinel.log_decision(date, regime_probs, brain.get_entropy(), holdings, equity=equity)

    # --- PHASE IV: PERFORMANCE ATTRIBUTION ---
    validator = StrategyValidator()
    # Validating against the operational timeline
    metrics = validator.calculate_metrics(equity_hist, timeline[250:])
    validator.print_terminal_report("ABMSM MASTER KERNEL", metrics, metrics, "USD")

if __name__ == "__main__":
    run_headless_audit()