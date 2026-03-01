import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class RiskConfig:
    target_volatility: float = 0.20  # Target 20% annualized portfolio vol
    max_position_leverage: float = 1.5 # Never buy more than 1.5x equity in one asset
    max_portfolio_leverage: float = 2.0 # Hard cap on total leverage
    stop_loss_atr_multiple: float = 3.0 # Default stop distance

class RiskManager:
    def __init__(self, config: RiskConfig, account_equity: float):
        self.config = config
        self.equity = account_equity
    
    def calculate_position_size(self, symbol, price, volatility):
        """
        Calculates how many shares to buy based on Volatility Targeting.
        
        Args:
            symbol (str): Ticker
            price (float): Current Price
            volatility (float): Annualized Volatility (e.g., 0.15 for 15%)
            
        Returns:
            dict: { 'shares': int, 'position_value': float, 'weight': float }
        """
        if volatility <= 0.01: # Avoid division by zero
            volatility = 0.01
            
        # 1. Volatility Targeting Formula
        # We want the position to contribute 'target_volatility' to the portfolio risk
        # Simplified: (Equity * Target_Vol) / Asset_Vol
        # Note: In a real portfolio, we divide Target_Vol by sqrt(N_Assets), 
        # but here we size each asset independently as if it's a silo.
        
        # We use a 'Volatility Factor' to scale. 
        # If Target=20%, Asset=10%, we leverage 2x.
        # If Target=20%, Asset=40%, we hold 0.5x.
        leverage_factor = self.config.target_volatility / volatility
        
        # 2. Cap Leverage
        leverage_factor = min(leverage_factor, self.config.max_position_leverage)
        
        # 3. Calculate Value and Shares
        target_position_value = self.equity * leverage_factor
        shares = int(target_position_value / price)
        
        return {
            'symbol': symbol,
            'shares': shares,
            'value': target_position_value,
            'leverage_applied': leverage_factor,
            'volatility_detected': volatility
        }

    def calculate_stop_loss(self, price, atr_value, direction="LONG"):
        """
        Calculates dynamic stop loss price.
        """
        dist = atr_value * self.config.stop_loss_atr_multiple
        
        if direction == "LONG":
            stop_price = price - dist
        else:
            stop_price = price + dist
            
        return stop_price

    def update_equity(self, new_equity):
        self.equity = new_equity