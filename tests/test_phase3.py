import pandas as pd
import numpy as np
from datetime import datetime
from src.signals import TradeSignal
from src.validator import SignalValidator

def test_validator():
    print("====== PHASE 3: VALIDATION LOGIC TEST ======")
    
    # 1. Setup Validator
    validator = SignalValidator()
    
    # 2. Create Mock Market Data (Simulating a single row from FeatureEngineer)
    # Scenario A: Good conditions (High Volume, Normal Volatility)
    good_market_data = pd.Series({
        'Rvol': 1.5,        # Strong Volume
        'Vol_ratio': 0.9,   # Compressing/Stable Volatility
        'Close': 450.0
    })

    # Scenario B: Dangerous conditions (Low Volume, Volatility Shock)
    bad_market_data = pd.Series({
        'Rvol': 0.5,        # Weak Volume
        'Vol_ratio': 1.8,   # Explosion!
        'Close': 450.0
    })

    # 3. Create a Dummy Signal
    signal = TradeSignal(
        symbol="SPY",
        action="LONG",
        strategy_name="TrendEngine",
        timestamp=datetime.now()
    )

    print("\n--- TEST 1: Good Data ---")
    accepted, reason = validator.validate(signal, good_market_data, regime_probs=[0.8, 0.1, 0.1, 0.0])
    print(f"Result: {'ACCEPTED' if accepted else 'REJECTED'}")
    print(f"Reason: {reason}")

    print("\n--- TEST 2: Bad Data (Should Fail) ---")
    accepted, reason = validator.validate(signal, bad_market_data, regime_probs=[0.8, 0.1, 0.1, 0.0])
    print(f"Result: {'ACCEPTED' if accepted else 'REJECTED'}")
    print(f"Reason: {reason}")
    
if __name__ == "__main__":
    test_validator()