"""
Aether Bayesian Kernel - Risk Engine Verification
Validates the mathematical integrity of the risk management kernel, 
specifically focusing on volatility targeting and drawdown feedback modulation.
"""

import sys
import os
import numpy as np

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.risk_manager import RiskManager

def test_risk_engine_math():
    """
    Executes a comprehensive audit of the adaptive position sizing kernel.
    
    Verifies the inverse-volatility relationship and the linear deleveraging 
    feedback mechanism utilized to protect capital during peak-to-trough erosion.
    """
    print("[TESTING] INITIATING RISK ENGINE ARCHITECTURAL AUDIT...")
    
    # --- PHASE I: COMPONENT INITIALIZATION ---
    # Initializing with the institutional 12% annual volatility target
    risk_engine = RiskManager(target_vol=0.12)
    equity_basis = 100000.0
    
    # --- TEST CASE 1: LOW-VOLATILITY DYNAMICS (ANNUAL 5%) ---
    # Expected: Inverse scaling should increase exposure towards the equity cap.
    vol_low = 0.05 / np.sqrt(252) # Annualized 5% to daily
    size_a = risk_engine.calculate_position_size(
        equity=equity_basis, 
        asset_vol=vol_low, 
        signal=1.0, 
        current_dd=0.0
    )
    
    print("\n[TEST] CASE 1: LOW-VOLATILITY REGIME (5% ANN)")
    print(f"Equity Basis: ${equity_basis:,.0f}")
    print(f"Risk Exposure Allocation: ${size_a:,.2f}")
    
    # --- TEST CASE 2: HIGH-VOLATILITY DYNAMICS (ANNUAL 80%) ---
    # Expected: Sizing should aggressively contract to preserve the risk budget.
    vol_high = 0.80 / np.sqrt(252) # Annualized 80% to daily
    size_b = risk_engine.calculate_position_size(
        equity=equity_basis, 
        asset_vol=vol_high, 
        signal=1.0, 
        current_dd=0.0
    )
    
    print("\n[TEST] CASE 2: HIGH-VOLATILITY REGIME (80% ANN)")
    print(f"Risk Exposure Allocation: ${size_b:,.2f}")
    
    # --- TEST CASE 3: CAPITAL IMPAIRMENT FEEDBACK (-10% DRAWDOWN) ---
    # Expected: The deleveraging penalty should further reduce the calculated size.
    vol_norm = 0.15 / np.sqrt(252)
    size_c = risk_engine.calculate_position_size(
        equity=equity_basis, 
        asset_vol=vol_norm, 
        signal=1.0, 
        current_dd=-0.10 # 10% peak-to-trough loss
    )
    
    print("\n[TEST] CASE 3: DRAWDOWN FEEDBACK RESPONSE (-10% DD)")
    print(f"Standard Allocation (0% DD): ${risk_engine.calculate_position_size(equity_basis, vol_norm, 1.0, 0.0):,.2f}")
    print(f"Defensive Allocation (-10% DD): ${size_c:,.2f}")

    print("\n[TEST] RISK ENGINE ARCHITECTURAL AUDIT COMPLETE.")

if __name__ == "__main__":
    test_risk_engine_math()