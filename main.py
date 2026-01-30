import sys
import pandas as pd
import numpy as np
import yfinance as yf
import yaml # Import the YAML library
from datetime import datetime

from src.features import FeatureEngineer
from src.regime import RegimeDetector
from src.risk.manager import RiskManager, RiskConfig
from src.portfolio import PortfolioController
from src.strategies.trend_engine import TrendEngine
from src.strategies.mean_reversion import MeanReversionEngine
from src.execution import PaperExecutionEngine

def load_config(path='config.yaml'):
    """Loads the system configuration from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_live_cycle(config):
    print(f"\n====== LIVE CYCLE START: {datetime.now()} ======")
    
    # --- Parameters are now loaded from config ---
    ticker_info = config['data']['universe'][0] # For V1, we only use the first asset
    TICKER = ticker_info['symbol']
    
    # 1. Load Live Data
    print(f"Fetching live data for {TICKER}...")
    df = yf.download(TICKER, period="2y", interval="1d", progress=False, auto_adjust=True)
    
    # 2. Engineer Features
    fe = FeatureEngineer(df)
    fe.add_volatility_features().add_trend_features().add_volume_features()
    data = fe.get_features()
    
    # 3. Initialize and Load/Train Regime Model
    regime_engine = RegimeDetector(config)
    try:
        # Try to load a pre-trained model first
        regime_engine.load()
    except FileNotFoundError:
        # If no model exists, train a new one and save it
        print("No pre-trained model found. Training a new one...")
        regime_engine.fit(data)

    # Predict TODAY'S Regime
    current_row = data.iloc[-1]
    regime_probs = regime_engine.predict_proba(pd.DataFrame([current_row]))[0]
    
    print(f"Detected Regime Probs: {regime_probs.round(2)}")
    
    # 4. Initialize Core Components from Config
    risk_config = RiskConfig(**config['risk']) # Unpack dict into dataclass
    risk_manager = RiskManager(risk_config, config['portfolio']['initial_capital'])
    engines = [TrendEngine(), MeanReversionEngine()]
    controller = PortfolioController(risk_manager, engines)
    
    # 5. Determine Allocations
    controller.update_allocations(regime_probs)
    print("Target Allocations:", {k: f"{v:.2%}" for k, v in controller.allocations.items()})
    
    # 6. Execute
    executor = PaperExecutionEngine(state_file=config['portfolio']['execution_state_file'])
    raw_price = df['Close'].iloc[-1]
    current_price = float(raw_price) if np.isscalar(raw_price) else float(raw_price.iloc[0])
    print(f"Market Price: ${current_price:.2f}")
    current_pos = executor.get_current_positions()
    equity = executor.positions['cash'] + (current_pos.get(TICKER, 0) * current_price)
    
    print(f"Current Equity: ${equity:,.2f}")
    
    executor.execute_rebalance(
        controller.allocations, 
        current_prices={TICKER: current_price}, 
        total_equity=equity
    )

if __name__ == "__main__":
    try:
        config = load_config()
        run_live_cycle(config)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()