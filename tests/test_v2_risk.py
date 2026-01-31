import sys
import os
import pandas as pd
import numpy as np

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.portfolio_v2 import PortfolioControllerV2

def test_correlation_penalty():
    print("--- Testing V2 Portfolio Risk Logic ---")
    
    controller = PortfolioControllerV2()
    
    # 1. SCENARIO A: Uncorrelated Assets (Diversified)
    # 60 days of random returns for 3 assets
    np.random.seed(42)
    df_uncorr = pd.DataFrame(np.random.normal(0, 0.01, (60, 3)), columns=['A', 'B', 'C'])
    
    penalty_A = controller.calculate_correlation_penalty(df_uncorr)
    print(f"\n[Scenario A] Diversified Portfolio penalty: {penalty_A:.2f}")
    # Expect: ~1.0 (No penalty)
    
    # 2. SCENARIO B: Highly Correlated Assets (Systemic Risk)
    # All assets move identical to 'A'
    df_corr = pd.DataFrame()
    df_corr['A'] = np.random.normal(0, 0.01, 60)
    df_corr['B'] = df_corr['A'] * 0.95 + np.random.normal(0, 0.001, 60) # Almost identical
    df_corr['C'] = df_corr['A'] * 0.90 + np.random.normal(0, 0.001, 60)
    
    penalty_B = controller.calculate_correlation_penalty(df_corr)
    print(f"\n[Scenario B] Systemic Risk Portfolio penalty: {penalty_B:.2f}")
    # Expect: < 0.8 (Significant Penalty)

if __name__ == "__main__":
    test_correlation_penalty()