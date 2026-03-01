import sys
import os
import yaml
import pandas as pd
from datetime import datetime

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.engine.data_loader import MultiAssetLoader
from src.brain.abmsm import ABMSM
from src.engine.portfolio import PortfolioControllerV2
from src.research.validator import StrategyValidator
# Import all strategies
from src.strategies.trend_engine import TrendEngine
from src.strategies.mean_reversion import MeanReversionEngine
from src.strategies.relative_strength import RelativeStrengthEngine
from src.strategies.volatility_breakout import VolatilityBreakoutEngine

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_headless_audit():
    """Runs a full system audit and prints an Investment Memorandum."""
    print(f"--- QUANT OS V2.0 KERNEL AUDIT: {datetime.now()} ---")
    config = load_config()
    
    # 1. Initialize Components
    loader = MultiAssetLoader('config.yaml')
    brain = ABMSM.load(config['regime_model']['abmsm']['save_path'])
    
    engines = [
        TrendEngine(), 
        MeanReversionEngine(), 
        RelativeStrengthEngine(top_n=1), 
        VolatilityBreakoutEngine()
    ]
    controller = PortfolioControllerV2('config.yaml', engines)
    
    # 2. Load Processed Data
    data_dict = loader.load_and_process_data()
    symbols = list(data_dict.keys())
    timeline = data_dict[symbols[0]].index.sort_values()
    
    # 3. Backtest Simulation (Headless)
    equity_hist = []
    initial_cap = config['portfolio']['initial_capital']
    equity = initial_cap
    cash = initial_cap
    holdings = {s: 0.0 for s in symbols}
    
    print(f"Simulating {len(timeline)} bars across {len(symbols)} assets...")
    
    for i in range(200, len(timeline)):
        date = timeline[i]
        market_slice = {s: df.loc[date] for s, df in data_dict.items() if date in df.index}
        
        # Brain Step
        primary = symbols[0]
        feats = [market_slice[primary]['Normalized_return'], market_slice[primary]['Atr']]
        regime_probs = brain.update(feats) # Uses the new brain.update logic
        
        # Portfolio Step
        hist_subset = pd.concat({s: df.iloc[i-60:i]['Normalized_return'] for s, df in data_dict.items()}, axis=1)
        allocs = controller.update_allocations(regime_probs, market_slice, hist_subset)
        
        # Mark to Market & Rebalance
        equity = cash + sum(holdings[s] * market_slice[s]['Close'] for s in holdings if s in market_slice)
        equity_hist.append(equity)
        
        new_cash = equity
        for s, weight in allocs.items():
            if s in market_slice:
                target_val = equity * weight
                holdings[s] = target_val / market_slice[s]['Close']
                new_cash -= target_val
        cash = new_cash

    # 4. Generate Tear Sheet
    validator = StrategyValidator()
    metrics = validator.calculate_metrics(equity_hist)
    validator.print_report("ABMSM ADAPTIVE CORE", metrics)

if __name__ == "__main__":
    run_headless_audit()