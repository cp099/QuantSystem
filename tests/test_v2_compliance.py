"""
Aether Bayesian Kernel - Compliance & Optimization Verification
Validates the functional and mathematical integrity of the Pre-Trade Compliance Gateway 
and the Mean-Variance Optimizer.
"""

import sys
import os
import pandas as pd
import numpy as np

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.compliance import PreTradeGateway
from src.engine.portfolio import CapitalAllocator
from src.strategies.trend_engine import TrendEngine

def test_pre_trade_gateway():
    """
    Validates SEC and SEBI compliance limits.
    """
    print("[TEST] INITIATING COMPLIANCE GATEWAY AUDIT...")
    
    # Init gateway with standard caps
    gate = PreTradeGateway(max_weight=0.30, min_equity_floor=70000.0, otr_limit=50.0)
    
    # 1. Test Concentration Capping
    proposed_allocs = {'AAPL': 0.50, 'MSFT': 0.20, 'GOOGL': 0.10}
    validated = gate.validate_allocations(proposed_allocs, current_equity=100000.0)
    
    # AAPL should be capped at 30%
    assert validated['AAPL'] == 0.30
    assert validated['MSFT'] == 0.20
    assert validated['GOOGL'] == 0.10
    
    # 2. Test SEC Capital protection emergency halt
    halted = gate.validate_allocations(proposed_allocs, current_equity=65000.0)
    assert all(w == 0.0 for w in halted.values())
    
    # 3. Test SEBI Price Band check
    # Proposed price is 11% change from previous close (breaches 9% safety band)
    assert gate.validate_price_band('RELIANCE.NS', 111.0, 100.0) is False
    # Proposed price is 5% change from previous close (compliant)
    assert gate.validate_price_band('RELIANCE.NS', 105.0, 100.0) is True
    
    # 4. Test OTR Compliance throttling
    gate.total_orders = 51
    gate.total_trades = 1
    assert gate.check_otr() is False
    
    print("[TEST] COMPLIANCE GATEWAY AUDIT COMPLETE.")

def test_mean_variance_optimization():
    """
    Validates the Mean-Variance Optimizer behavior.
    """
    print("[TEST] INITIATING MEAN-VARIANCE OPTIMIZER AUDIT...")
    
    engines = [TrendEngine()]
    allocator = CapitalAllocator('config.yaml', engines)
    
    # Construct a historical dataframe with high correlation between asset A and B
    np.random.seed(42)
    history_df = pd.DataFrame()
    history_df['AAPL'] = np.random.normal(0.001, 0.01, 100)
    history_df['MSFT'] = history_df['AAPL'] * 0.95 + np.random.normal(0, 0.001, 100) # High correlation
    history_df['GLD'] = np.random.normal(-0.001, 0.01, 100)                        # Uncorrelated
    
    # Proposed signals: all active
    market_slice = {
        'AAPL': pd.Series({'Close': 150, 'Momentum': 0.05, 'v': 2.0}),
        'MSFT': pd.Series({'Close': 300, 'Momentum': 0.05, 'v': 2.0}),
        'GLD': pd.Series({'Close': 180, 'Momentum': 0.05, 'v': 1.5})
    }
    
    regime_bull = [0.8, 0.0, 0.0, 0.0, 0.2] # TrendEngine is active
    
    # Run allocation with correlation data
    allocs = allocator.update_allocations(regime_bull, market_slice, history_df)
    
    print("\n[MVO Test] Allocations:")
    print(allocs)
    
    # MVO should diversify: because AAPL and MSFT are highly correlated, 
    # and GLD is uncorrelated and has positive momentum, GLD should get a significant weight.
    assert allocs['GLD'] > 0.0
    assert allocs['AAPL'] > 0.0
    assert allocs['MSFT'] > 0.0
    
    # Total allocations should respect the 1.0x equity cap and the 0.50 budget (since Trend Engine gets 0.50 budget)
    # Wait, the budget for Trend Engine is 0.50. So total allocation should be capped at 0.50!
    assert sum(allocs.values()) <= 0.50 + 1e-6
    
    print("[TEST] MEAN-VARIANCE OPTIMIZER AUDIT COMPLETE.")
