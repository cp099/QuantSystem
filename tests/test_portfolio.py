"""
Aether Bayesian Kernel - Allocation Unit Testing
Validates the hierarchical capital budgeting logic within the Capital Allocator, 
ensuring accurate distribution across expansionary and stationary market regimes.
"""

import sys
import os
import yaml

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.trend_engine import TrendEngine
from src.strategies.mean_reversion import MeanReversionEngine
from src.strategies.relative_strength import RelativeStrengthEngine
from src.engine.risk_manager import RiskManager
from src.engine.portfolio import CapitalAllocator

def test_allocation_logic():
    """
    Executes a series of stress tests on the capital distribution kernel.
    
    Verifies that the allocator correctly identifies dominant Bayesian states 
    and shifts capital budgets between Trend and Mean Reversion specialists 
    while maintaining institutional risk constraints.
    """
    print("[TESTING] INITIATING ALLOCATION KERNEL AUDIT...")
    
    # --- PHASE I: COMPONENT INITIALIZATION ---
    risk = RiskManager(target_vol=0.12)
    engines = [
        TrendEngine(), 
        MeanReversionEngine(), 
        RelativeStrengthEngine(top_n=1)
    ]
    
    # Allocator requires the path to system configuration
    controller = CapitalAllocator('config.yaml', engines)
    
    # Constructing a realistic cross-sectional mock market slice
    import pandas as pd
    market_slice = {
        'AAPL': pd.Series({'Close': 150.0, 'Momentum': 0.05, 'Normalized_return': -2.0, 'v': 2.0}),
        'MSFT': pd.Series({'Close': 300.0, 'Momentum': -0.01, 'Normalized_return': 0.0, 'v': 0.5})
    }
    
    # --- TEST CASE 1: EXPANSIONARY DYNAMICS (STATE 0 & 4) ---
    # State Vector: [Bull=0.8, Bear=0.0, Grind=0.0, Crash=0.0, Recov=0.2]
    regime_bull = [0.8, 0.0, 0.0, 0.0, 0.2]
    controller.update_allocations(regime_bull, market_slice, None)
    
    print("\n[TEST] CASE 1: EXPANSIONARY DYNAMICS")
    print(f"AAPL Allocation: {controller.allocations.get('AAPL', 0):.2%}")
    print(f"MSFT Allocation: {controller.allocations.get('MSFT', 0):.2%}")
    # Expected: Trend (0.4) + RS (0.3) = 0.70 (DL gets 0.3 but is not in engines list)
    assert abs(controller.allocations.get('AAPL', 0) - 0.70) < 1e-6
    assert controller.allocations.get('MSFT', 0) == 0.0
    
    # --- TEST CASE 2: STATIONARY DYNAMICS (STATE 1 & 2) ---
    # State Vector: [Bull=0.0, Bear=0.1, Grind=0.8, Crash=0.0, Recov=0.1]
    regime_chop = [0.0, 0.1, 0.8, 0.0, 0.1]
    controller.update_allocations(regime_chop, market_slice, None)
    
    print("\n[TEST] CASE 2: STATIONARY DYNAMICS")
    print(f"AAPL Allocation: {controller.allocations.get('AAPL', 0):.2%}")
    print(f"MSFT Allocation: {controller.allocations.get('MSFT', 0):.2%}")
    # Expected: MR (0.45) + RS (0.03) = 0.48 (Trend gated, DL budget not in engines list)
    assert abs(controller.allocations.get('AAPL', 0) - 0.48) < 1e-6
    assert controller.allocations.get('MSFT', 0) == 0.0
 
    # --- TEST CASE 3: CONTRACTIONARY DYNAMICS (STATE 3) ---
    # State Vector: [Bull=0.0, Bear=0.0, Grind=0.0, Crash=0.9, Recov=0.1]
    regime_crash = [0.0, 0.0, 0.0, 0.9, 0.1]
    controller.update_allocations(regime_crash, market_slice, None)
    
    # Total assigned budget should be low, effectively raising cash levels
    total_budget = sum(controller.allocations.values())
    
    print("\n[TEST] CASE 3: CONTRACTIONARY DYNAMICS")
    print(f"Aggregate Exposure: {total_budget:.2%}")
    print(f"Defensive Cash Position: {1.0 - total_budget:.2%}")
    # Expected: 0.0 allocation since all engines are gated during crash
    assert controller.allocations.get('AAPL', 0) == 0.0
    assert total_budget == 0.0
 
    print("\n[TEST] ALLOCATION KERNEL AUDIT COMPLETE.")

if __name__ == "__main__":
    test_allocation_logic()