import sys
import pandas as pd
import numpy as np # Added numpy for type safety
import yfinance as yf
from datetime import datetime
from src.features import FeatureEngineer
from src.regime import RegimeDetector
from src.risk.manager import RiskManager, RiskConfig
from src.portfolio import PortfolioController
from src.strategies.trend_engine import TrendEngine
from src.strategies.mean_reversion import MeanReversionEngine
from src.execution import PaperExecutionEngine

# SETUP
TICKER = "SPY"
CAPITAL = 100000

def run_live_cycle():
    print(f"\n====== LIVE CYCLE START: {datetime.now()} ======")
    
    # 1. Load Live Data
    print(f"Fetching live data for {TICKER}...")
    df = yf.download(TICKER, period="2y", interval="1d", progress=False, auto_adjust=True)
    
    # 2. Engineer Features
    fe = FeatureEngineer(df)
    fe.add_volatility_features().add_trend_features().add_volume_features()
    data = fe.get_features()
    
    # 3. Load/Train Regime Model
    regime_engine = RegimeDetector(n_components=4)
    regime_engine.feature_cols = ['Vol_ratio', 'Momentum', 'Vol_short']
    regime_engine.fit(data)
    
    # Predict TODAY'S Regime
    current_row = data.iloc[-1]
    # Reshape and predict
    regime_probs = regime_engine.predict_proba(pd.DataFrame([current_row]))[0]
    current_regime = regime_probs.argmax()
    
    print(f"Detected Regime: {current_regime} (Probs: {regime_probs.round(2)})")
    
    # 4. Initialize Engines
    risk_manager = RiskManager(RiskConfig(), CAPITAL)
    engines = [TrendEngine(), MeanReversionEngine()]
    controller = PortfolioController(risk_manager, engines)
    
    # 5. Determine Allocations
    controller.update_allocations(regime_probs)
    print("Target Allocations:", {k: round(float(v), 2) for k, v in controller.allocations.items()})
    
    # 6. Execute
    executor = PaperExecutionEngine()
    
    # --- FIX IS HERE: Force conversion to standard float ---
    raw_price = df['Close'].iloc[-1]
    if isinstance(raw_price, pd.Series):
        current_price = float(raw_price.iloc[0])
    else:
        current_price = float(raw_price)
        
    print(f"Market Price: ${current_price:.2f}")

    # Calculate Total Equity
    current_pos = executor.get_current_positions()
    equity = executor.positions['cash'] + (current_pos.get('SPY', 0) * current_price)
    
    print(f"Current Equity: ${equity:,.2f}")
    
    executor.execute_rebalance(
        controller.allocations, 
        current_prices={'SPY': current_price}, 
        total_equity=equity
    )

if __name__ == "__main__":
    try:
        run_live_cycle()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()