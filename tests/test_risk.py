import sys
import os
# Fix path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk.manager import RiskManager, RiskConfig

def test_risk_math():
    print("====== PHASE 4: RISK MANAGEMENT TEST ======")
    
    # 1. Setup (Account with $100,000)
    config = RiskConfig(target_volatility=0.20) # We want 20% risk
    risk_engine = RiskManager(config, account_equity=100000)
    
    current_price = 100.0
    
    # --- SCENARIO A: Low Volatility Asset (e.g., Bonds, 5% Vol) ---
    vol_low = 0.05 
    size_a = risk_engine.calculate_position_size("BOND", current_price, vol_low)
    
    print(f"\n[Scenario A] Low Vol ({vol_low*100}%):")
    print(f"Leverage: {size_a['leverage_applied']}x")
    print(f"Shares: {size_a['shares']}")
    print(f"Value: ${size_a['value']:,.2f}")
    
    # Logic check: If vol is 5% and target is 20%, we should leverage 4x, 
    # BUT our max_position_leverage is 1.5x (in config).
    # So expected leverage is 1.5x.
    
    # --- SCENARIO B: High Volatility Asset (e.g., Crypto, 80% Vol) ---
    vol_high = 0.80
    size_b = risk_engine.calculate_position_size("COIN", current_price, vol_high)
    
    print(f"\n[Scenario B] High Vol ({vol_high*100}%):")
    print(f"Leverage: {size_b['leverage_applied']}x")
    print(f"Shares: {size_b['shares']}")
    print(f"Value: ${size_b['value']:,.2f}")
    
    # Logic Check: Target 20% / Asset 80% = 0.25x Leverage.
    # Value should be $25,000.
    
    # --- SCENARIO C: Stop Loss Calc ---
    atr = 2.0
    stop = risk_engine.calculate_stop_loss(current_price, atr, "LONG")
    print(f"\n[Scenario C] Stop Loss (ATR {atr}):")
    print(f"Entry: {current_price}, Stop: {stop}")
    
if __name__ == "__main__":
    test_risk_math()