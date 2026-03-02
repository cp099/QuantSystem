"""
Aether Bayesian Kernel - Protocol Unit Testing
Validates the integrity of the signal encapsulation protocol and 
cross-modular data structures.
"""

import sys
import os
from datetime import datetime

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.signals import TradeSignal

def test_signal_encapsulation():
    """
    Verifies the metadata integrity of the TradeSignal container.
    """
    print("[TESTING] INITIATING SIGNAL PROTOCOL AUDIT...")
    
    # Constructing an institutional conviction signal
    signal = TradeSignal(
        symbol="RELIANCE.NS",
        action="LONG",
        strategy_name="Trend_Persistence",
        timestamp=datetime.now(),
        strength=0.85,
        comment="Alpha-Reflexive Entry"
    )
    
    # Validating data attributes
    assert signal.symbol == "RELIANCE.NS"
    assert signal.strength == 0.85
    
    print(f"[TEST] PROTOCOL ENCAPSULATION VERIFIED: {signal.symbol} | {signal.action}")
    print("[TESTING] SIGNAL PROTOCOL AUDIT COMPLETE.")

if __name__ == "__main__":
    test_signal_encapsulation()