import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.trend_engine import TrendEngine
from src.strategies.mean_reversion import MeanReversionEngine
from src.risk.manager import RiskManager, RiskConfig
from src.portfolio import PortfolioController

def test_allocation_logic():
    print("====== PHASE 5: PORTFOLIO CONTROLLER TEST ======")
    
    # 1. Setup
    risk = RiskManager(RiskConfig(), 100000)
    engines = [TrendEngine(), MeanReversionEngine()]
    controller = PortfolioController(risk, engines)
    
    # --- SCENARIO 1: Strong Bull Market (Regime 0) ---
    # Probs: [Bull=0.9, Crash=0.0, Rec=0.1, Chop=0.0]
    regime_bull = [0.9, 0.0, 0.1, 0.0]
    controller.update_allocations(regime_bull)
    
    print("\n[Scenario 1] Bull Market:")
    print(f"Trend Alloc: {controller.get_allocation('Trend_Engine'):.2%}")
    print(f"MeanRev Alloc: {controller.get_allocation('Mean_Reversion'):.2%}")
    # Expect Trend to be dominant (~95%)
    
    # --- SCENARIO 2: Choppy Market (Regime 3) ---
    # Probs: [Bull=0.0, Crash=0.0, Rec=0.1, Chop=0.9]
    regime_chop = [0.0, 0.0, 0.1, 0.9]
    controller.update_allocations(regime_chop)
    
    print("\n[Scenario 2] Choppy Market:")
    print(f"Trend Alloc: {controller.get_allocation('Trend_Engine'):.2%}")
    print(f"MeanRev Alloc: {controller.get_allocation('Mean_Reversion'):.2%}")
    # Expect MeanRev to be dominant (~95%)

    # --- SCENARIO 3: CRASH (Regime 1) ---
    # Probs: [Bull=0.0, Crash=0.8, Rec=0.0, Chop=0.2]
    regime_crash = [0.0, 0.8, 0.0, 0.2]
    controller.update_allocations(regime_crash)
    
    print("\n[Scenario 3] Market Crash:")
    print(f"Trend Alloc: {controller.get_allocation('Trend_Engine'):.2%}")
    print(f"MeanRev Alloc: {controller.get_allocation('Mean_Reversion'):.2%}")
    print(f"Cash/Unused: {1.0 - sum(controller.allocations.values()):.2%}")
    # Expect mostly Cash (~80%)

if __name__ == "__main__":
    test_allocation_logic()