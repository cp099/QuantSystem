"""
Aether Bayesian Kernel - Risk Concentration Testing
Validates the mathematical engine responsible for detecting systemic risk 
clustering through eigenvalue decomposition of the correlation matrix.
"""

import sys
import os
import pandas as pd
import numpy as np

# Environment configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.portfolio import CapitalAllocator

def test_concentration_penalty():
    """
    Executes a statistical audit of the risk concentration kernel.
    """
    print("[TESTING] INITIATING RISK CONCENTRATION AUDIT...")
    
    # Initialize with empty engines for pure math validation
    allocator = CapitalAllocator('config.yaml', engines=[])
    
    # --- TEST CASE 1: STOCHASTIC DIVERSIFICATION ---
    np.random.seed(42)
    df_uncorr = pd.DataFrame(np.random.normal(0, 0.01, (60, 3)), columns=['A', 'B', 'C'])
    
    penalty_a = allocator.calculate_correlation_penalty(df_uncorr)
    print(f"[TEST] CASE 1 MULTIPLIER: {penalty_a:.2f}x")
    assert penalty_a == 1.0 # Should not penalize diversified data
    
    # --- TEST CASE 2: SYSTEMIC RISK CLUSTERING ---
    df_corr = pd.DataFrame()
    df_corr['A'] = np.random.normal(0, 0.01, 60)
    df_corr['B'] = df_corr['A'] * 0.98 + np.random.normal(0, 0.001, 60) # High correlation
    df_corr['C'] = df_corr['A'] * 0.95 + np.random.normal(0, 0.001, 60)
    
    penalty_b = allocator.calculate_correlation_penalty(df_corr)
    print(f"[TEST] CASE 2 MULTIPLIER: {penalty_b:.2f}x")
    assert penalty_b < 1.0 # Should apply penalty for clustering
    
    print("[TEST] RISK CONCENTRATION AUDIT COMPLETE.")

if __name__ == "__main__":
    test_concentration_penalty()