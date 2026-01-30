import pandas as pd
import json
import os
from datetime import datetime

class PaperExecutionEngine:
    def __init__(self, state_file="data/processed/portfolio_state.json"):
        self.state_file = state_file
        self.positions = self.load_state()
        
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"cash": 100000.0, "holdings": {}}

    def save_state(self):
        # Create directory if not exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.positions, f, indent=4)

    def get_current_positions(self):
        return self.positions['holdings']

    def execute_rebalance(self, target_allocations, current_prices, total_equity):
        """
        Calculates orders to match target allocations.
        target_allocations: {'Trend_Engine': 0.4, ...} -> We need to map this to assets.
        For Phase 7, we simplify: Trend Engine = SPY, Mean Reversion = SPY.
        """
        print(f"\n[{datetime.now()}] --- EXECUTING REBALANCE ---")
        
        # 1. Calculate Net Target Weight for SPY
        # (Both engines trade SPY in our current setup)
        net_spy_weight = 0.0
        for engine, weight in target_allocations.items():
            # In a complex system, engines would return specific tickers.
            # Here, both engines trade the main asset.
            # We assume the engines are fully invested if active (Signal=1)
            # But the PortfolioController already weighted them.
            net_spy_weight += weight

        print(f"Target Net Exposure: {net_spy_weight:.2%}")

        # 2. Calculate Shares Needed
        price = current_prices['SPY']
        target_value = total_equity * net_spy_weight
        target_shares = int(target_value / price)
        
        current_shares = self.positions['holdings'].get('SPY', 0)
        
        diff = target_shares - current_shares
        
        if diff == 0:
            print("No Trade Required.")
            return

        # 3. 'Execute' Trade
        action = "BUY" if diff > 0 else "SELL"
        cost = diff * price
        
        print(f"ORDER GENERATED: {action} {abs(diff)} SPY @ ${price:.2f}")
        print(f"Value: ${abs(cost):,.2f}")
        
        # 4. Update Internal State
        self.positions['cash'] -= cost
        self.positions['holdings']['SPY'] = target_shares
        self.save_state()
        
        print(f"New Portfolio: Cash=${self.positions['cash']:.0f}, SPY={target_shares} shares")