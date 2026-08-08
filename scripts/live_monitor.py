"""
Aether Bayesian Kernel - Live Market Observation & Paper Trading Monitor
Consumes real-time global market data streams, estimates latent regimes,
generates MLP convictions, checks compliance constraints, and triggers Buy/Sell markers.
"""

import sys
import os
import argparse
import yaml
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.data_loader import UniversalLoader
from src.brain.abmsm import ABMSM
from src.engine.compliance import PreTradeGateway
from src.strategies.deep_learning import DeepLearningEngine

def is_market_open(ticker):
    """
    Evaluates whether the target asset class is currently trading.
    """
    # 1. Cryptocurrencies (Open 24/7)
    if ticker.endswith("-USD"):
        return True, "MARKET OPEN (24/7 CRYPTO)"
        
    # 2. Forex (Open Mon-Fri 24h)
    if ticker.endswith("=X"):
        now_utc = datetime.utcnow()
        if now_utc.weekday() >= 5: # Sat or Sun
            return False, "MARKET CLOSED (FOREX WEEKEND)"
        return True, "MARKET OPEN (24h FOREX)"
        
    # 3. Indian Equities (NSE/BSE)
    # Open Mon-Fri, 9:15 AM to 3:30 PM IST (3:45 AM to 10:00 AM UTC)
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        now_utc = datetime.utcnow()
        if now_utc.weekday() >= 5:
            return False, "MARKET CLOSED (NSE WEEKEND)"
        
        current_time_utc = now_utc.time()
        start_time = datetime.strptime("03:45", "%H:%M").time()
        end_time = datetime.strptime("10:00", "%H:%M").time()
        if start_time <= current_time_utc <= end_time:
            return True, "MARKET OPEN (NSE LIVE)"
        return False, "MARKET CLOSED (NSE HOURS: 9:15 - 15:30 IST)"
        
    # 4. US Equities (NYSE/NASDAQ)
    # Open Mon-Fri, 9:30 AM to 4:00 PM EST (approx 13:30 - 21:00 UTC)
    now_utc = datetime.utcnow()
    if now_utc.weekday() >= 5:
        return False, "MARKET CLOSED (US WEEKEND)"
    
    current_time_utc = now_utc.time()
    start_time = datetime.strptime("13:30", "%H:%M").time()
    end_time = datetime.strptime("21:00", "%H:%M").time()
    if start_time <= current_time_utc <= end_time:
        return True, "MARKET OPEN (US LIVE)"
    return False, "MARKET CLOSED (US HOURS: 9:30 - 16:00 EST)"

def run_live_monitor(ticker, interval=10, allocation=100000.0):
    print(f"\n[LIVE MONITOR] INITIALIZING PIPELINE FOR: {ticker}")
    
    # Check Market Hours
    market_open, market_status_msg = is_market_open(ticker)
    if not market_open:
        print(f"[LIVE MONITOR HOURS WARNING] {market_status_msg}")
        print("[LIVE MONITOR HOURS WARNING] RUNNING IN CLOSED MARKET BACK-FILL MODE USING LAST KNOWN VALUES.")
    else:
        print(f"[LIVE MONITOR STATUS] {market_status_msg}")
    
    # 1. Load Configurations
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    loader = UniversalLoader()
    
    # Load Models
    try:
        brain = ABMSM.load(config['regime_model']['abmsm']['save_path'])
    except Exception as e:
        print(f"[LIVE MONITOR] ERROR loading ABMSM brain: {e}")
        return
        
    mlp_engine = DeepLearningEngine(model_path="models/global_neural_network.pkl")
    if mlp_engine.model is None:
        print("[LIVE MONITOR] ERROR: Deep Learning MLP model not loaded.")
        return
        
    compliance_gate = PreTradeGateway(max_weight=0.30, min_equity_floor=70000.0)
    
    # Paper Trading Account State with dynamic allocation starting budget
    initial_cap = allocation
    cash = initial_cap
    shares = 0.0
    equity = initial_cap
    
    # Clear previous live audit log for a clean UI run
    live_log_path = "logs/audit/live_audit.jsonl"
    os.makedirs(os.path.dirname(live_log_path), exist_ok=True)
    if os.path.exists(live_log_path):
        try:
            os.remove(live_log_path)
        except Exception:
            pass
            
    print(f"[LIVE MONITOR] PAPER ACCOUNT STANDING: Cash=${cash:,.2f}, Equity=${equity:,.2f}")
    print(f"[LIVE MONITOR] STREAM ACTIVE (POLLING EVERY {interval} SECONDS)... PRESS CTRL+C TO STOP")
    
    features = ['v', 'r', 'c', 'a', 'd', 'l', 'b']
    
    try:
        while True:
            # A. Fetch data
            try:
                # 3-year period is highly optimized for fast queries while preserving HMM IQR scaling window
                df, bench_df, local_ccy = loader.fetch_and_engineer(ticker, period="3y")
            except Exception as e:
                print(f"[LIVE MONITOR NETWORK ERROR] Could not fetch real-time data: {e}")
                time.sleep(interval)
                continue
                
            if df.empty:
                print("[LIVE MONITOR DATA WARNING] Received empty dataset. Retrying...")
                time.sleep(interval)
                continue
                
            # Extract latest bar
            row = df.iloc[-1]
            current_price = float(row['Close'])
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract features for brain and MLP
            feats = [float(row[f]) for f in features]
            
            # B. State Space Propagation
            regime_probs = brain.update(feats, adapt=True)
            dominant_regime = int(regime_probs.argmax())
            regime_confidence = float(regime_probs.max())
            entropy = float(brain.get_entropy())
            
            # C. MLP Conviction Upwards Probability
            market_slice = {ticker: row}
            mlp_signals = mlp_engine.generate_signals(market_slice, regime_probs)
            mlp_conviction = mlp_signals.get(ticker, 0.5)
            
            # D. Portfolio Allocation Decisions & Compliance Gates
            # Mark-to-market
            equity = cash + (shares * current_price)
            
            # Default allocations: Buy up to 30% on high conviction, Liquidate on low
            target_alloc = 0.0
            signal_type = "HOLD"
            
            if mlp_conviction > 0.53:
                target_alloc = 0.30
                signal_type = "BUY"
            elif mlp_conviction < 0.47:
                target_alloc = 0.0
                signal_type = "SELL"
            else:
                # Maintain current weight allocation
                target_alloc = (shares * current_price) / equity if equity > 0 else 0.0
                signal_type = "HOLD"
                
            # Validate allocations via compliance access gate
            validated_allocs = compliance_gate.validate_allocations({ticker: target_alloc}, equity)
            valid_weight = validated_allocs.get(ticker, 0.0)
            
            # Capital realignment
            target_val = equity * valid_weight
            current_val = shares * current_price
            trade_val = target_val - current_val
            
            # Transaction friction controls
            if abs(trade_val) > 0.02 * equity:
                compliance_gate.total_orders += 1
                if compliance_gate.check_otr():
                    pos = df.index.get_loc(df.index[-1])
                    prev_close = df['Close'].iloc[pos-1] if pos > 0 else current_price
                    
                    if compliance_gate.validate_price_band(ticker, current_price, prev_close):
                        compliance_gate.total_trades += 1
                        
                        vol = float(row['vol_pct'])
                        slippage = abs(trade_val) * (0.05 * vol)
                        brokerage = abs(trade_val) * 0.0005
                        friction = slippage + brokerage
                        
                        cash -= (trade_val + friction)
                        shares = target_val / current_price
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] EXECUTION: {signal_type} {ticker} @ {current_price:.2f} (Units: {shares:.2f}, Equity: {local_ccy} {equity:.2f})")
                    else:
                        signal_type = "CIRCUIT_BLOCKED"
                else:
                    signal_type = "OTR_THROTTLED"
            else:
                signal_type = "HOLD"
                
            # Update equity post-rebalance
            equity = cash + (shares * current_price)
            
            # E. Audit Persistence Schema
            audit_entry = {
                "timestamp": date_str,
                "price": current_price,
                "features": {f: float(row[f]) for f in features},
                "dominant_regime": dominant_regime,
                "regime_confidence": regime_confidence,
                "entropy": entropy,
                "mlp_conviction": mlp_conviction,
                "signal": signal_type,
                "holdings": float(shares),
                "equity": float(equity),
                "cash": float(cash),
                "market_status": market_status_msg,
                "currency": local_ccy
            }
            
            with open(live_log_path, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n[LIVE MONITOR] Stream interrupted. Securing paper account balance.")
        print(f"[LIVE MONITOR] FINAL VALUE: Cash=${cash:,.2f}, Shares={shares:.2f}, Equity=${equity:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None, help="Security ticker to monitor")
    parser.add_argument("--interval", type=int, default=None, help="Polling interval in seconds")
    parser.add_argument("--allocation", type=float, default=None, help="Starting allocation fund pool")
    args = parser.parse_args()
    
    ticker = args.ticker or os.environ.get("ABK_TICKER") or "AAPL"
    
    interval_val = args.interval
    if interval_val is None:
        env_interval = os.environ.get("ABK_INTERVAL")
        interval_val = int(env_interval) if env_interval else 10
        
    allocation_val = args.allocation
    if allocation_val is None:
        env_alloc = os.environ.get("ABK_ALLOCATION")
        allocation_val = float(env_alloc) if env_alloc else 100000.0
        
    run_live_monitor(ticker, interval_val, allocation_val)
