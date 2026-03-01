import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.features import FeatureEngineer

class BacktestEngine:
    def __init__(self, controller, regime_engine, initial_capital=100000):
        self.controller = controller
        self.regime_engine = regime_engine
        self.equity = initial_capital
        self.cash = initial_capital
        self.holdings = {} # {symbol: {'shares': 0, 'value': 0}}
        self.history = [] # For logging daily stats

    def run(self, data_dict):
        """
        Runs the simulation.
        data_dict: { 'SPY': dataframe, 'TLT': dataframe, ... }
        """
        # 1. Align Data (Find common index)
        # We drive the simulation using SPY's index
        main_asset = list(data_dict.keys())[0]
        timeline = data_dict[main_asset].index
        
        print(f"Starting Backtest on {len(timeline)} bars...")
        
        for date in timeline:
            # --- MORNING SEQUENCE ---
            
            # A. Get Market State (using SPY as the 'Market' proxy for Regime)
            # In production, you might use a basket, but SPY is fine for Phase 6.
            spy_row = data_dict['SPY'].loc[date]
            
            # We need the features for this specific date
            # (In a real loop, we'd calc this incrementally, but for speed we pre-calc)
            # We assume features are already in the dataframe passed in.
            
            # Predict Regime
            # We reshape to 2D array because sklearn expects [[feat1, feat2...]]
            regime_feats = spy_row[['Vol_ratio', 'Momentum', 'Vol_short']].values.reshape(1, -1)
            regime_probs = self.regime_engine.predict_proba(pd.DataFrame([spy_row]))[0]
            current_regime = np.argmax(regime_probs)
            
            # B. Update Portfolio Allocation
            self.controller.update_allocations(regime_probs)
            
            # --- TRADING SEQUENCE ---
            daily_value = self.cash
            
            for symbol, df in data_dict.items():
                if date not in df.index: continue
                
                row = df.loc[date]
                price = row['Close']
                
                # 1. Ask Strategies for Signals
                # We sum signals from all engines weighted by allocation
                net_signal = 0
                
                for engine in self.controller.engines:
                    weight = self.controller.get_allocation(engine.name)
                    if weight > 0:
                        sig = engine.generate_signal(row, current_regime)
                        # Signal usually 1 or 0.
                        net_signal += (sig * weight)
                
                # net_signal is now the Target % of Equity for this asset
                # e.g., 0.45 means "Put 45% of equity into this asset"
                
                # 2. Risk Management (Sizing)
                # We use the RiskManager to cap this based on volatility
                # Note: Our RiskManager calculates distinct share counts. 
                # Here we simplify: The net_signal IS the target weight, 
                # but we scale it by Volatility using the RiskManager's logic if needed.
                # For this Phase, we trust the PortfolioController's weight output directly.
                
                target_value = self.equity * net_signal
                current_shares = self.holdings.get(symbol, {}).get('shares', 0)
                current_pos_value = current_shares * price
                
                # Rebalance if diff is significant (> 5% change)
                if abs(target_value - current_pos_value) > (self.equity * 0.05):
                    # Execute Trade
                    new_shares = int(target_value / price)
                    self.holdings[symbol] = {'shares': new_shares, 'value': target_value}
                    
                    # Update Cash (Cost of trade)
                    cost = (new_shares - current_shares) * price
                    self.cash -= cost
            
            # --- EVENING SEQUENCE ---
            # Mark to Market
            total_holdings_val = 0
            for sym in self.holdings:
                if date in data_dict[sym].index:
                    px = data_dict[sym].loc[date]['Close']
                    val = self.holdings[sym]['shares'] * px
                    self.holdings[sym]['value'] = val
                    total_holdings_val += val
            
            self.equity = self.cash + total_holdings_val
            
            # Log Logic
            self.history.append({
                'Date': date,
                'Equity': self.equity,
                'Cash': self.cash,
                'Regime': current_regime,
                'Allocation_Trend': self.controller.get_allocation('Trend_Engine'),
                'Allocation_MeanRev': self.controller.get_allocation('Mean_Reversion')
            })

        print(f"Backtest Complete. Final Equity: ${self.equity:,.2f}")
        return pd.DataFrame(self.history).set_index('Date')