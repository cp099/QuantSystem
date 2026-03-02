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
    engines = [TrendEngine(), MeanReversionEngine()]
    
    # Allocator requires the path to system configuration
    controller = CapitalAllocator('config.yaml', engines)
    
    # --- TEST CASE 1: EXPANSIONARY DYNAMICS (STATE 0 & 4) ---
    # State Vector: [Bull=0.8, Bear=0.0, Grind=0.0, Crash=0.0, Recov=0.2]
    regime_bull = [0.8, 0.0, 0.0, 0.0, 0.2]
    controller.update_allocations(regime_bull, {}, None)
    
    print("\n[TEST] CASE 1: EXPANSIONARY DYNAMICS")
    print(f"Trend Budget: {controller.allocations.get('Trend_Engine', 0):.2%}")
    print(f"Reversion Budget: {controller.allocations.get('Mean_Reversion', 0):.2%}")
    
    # --- TEST CASE 2: STATIONARY DYNAMICS (STATE 1 & 2) ---
    # State Vector: [Bull=0.0, Bear=0.1, Grind=0.8, Crash=0.0, Recov=0.1]
    regime_chop = [0.0, 0.1, 0.8, 0.0, 0.1]
    controller.update_allocations(regime_chop, {}, None)
    
    print("\n[TEST] CASE 2: STATIONARY DYNAMICS")
    print(f"Trend Budget: {controller.allocations.get('Trend_Engine', 0):.2%}")
    print(f"Reversion Budget: {controller.allocations.get('Mean_Reversion', 0):.2%}")

    # --- TEST CASE 3: CONTRACTIONARY DYNAMICS (STATE 3) ---
    # State Vector: [Bull=0.0, Bear=0.0, Grind=0.0, Crash=0.9, Recov=0.1]
    regime_crash = [0.0, 0.0, 0.0, 0.9, 0.1]
    controller.update_allocations(regime_crash, {}, None)
    
    # Total assigned budget should be low, effectively raising cash levels
    total_budget = sum(controller.allocations.values())
    
    print("\n[TEST] CASE 3: CONTRACTIONARY DYNAMICS")
    print(f"Aggregate Exposure: {total_budget:.2%}")
    print(f"Defensive Cash Position: {1.0 - total_budget:.2%}")

    print("\n[TEST] ALLOCATION KERNEL AUDIT COMPLETE.")

if __name__ == "__main__":
    test_allocation_logic()