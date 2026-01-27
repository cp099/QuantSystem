import pandas as pd
import numpy as np
from .base import BaseStrategy

class TrendEngine(BaseStrategy):
    def __init__(self, lookback=20):
        super().__init__("Trend_Follower_V1")
        self.lookback = lookback
        
    def generate_signal(self, market_data, regime_probs):
        """
        Logic: Donchian Channel Breakout
        Constraint: Only fires if Regime is Bullish (0) or Volatile Bull (2).
        """
        # 1. Regime Filter (The Gatekeeper)
        # 0=Steady, 1=Crash, 2=Recov, 3=Chop
        prob_bullish = regime_probs[0] + regime_probs[2] 
        
        if prob_bullish < 0.5:
            return {'action': 'HOLD', 'reason': 'Regime Incompatible'}

        # 2. Strategy Logic
        current_price = market_data['Close'].iloc[-1]
        high_n = market_data['High'].rolling(self.lookback).max().iloc[-2] # Previous N bars
        low_n = market_data['Low'].rolling(self.lookback).min().iloc[-2]
        atr = self.get_atr(market_data)

        # ENTRY: Breakout of N-day high
        if self.position == 0:
            if current_price > high_n:
                self.stop_loss = current_price - (2.5 * atr) # Trailing stop
                return {'action': 'BUY', 'stop': self.stop_loss}
        
        # EXIT: Breakdown of N-day low
        elif self.position == 1:
            # Trailing Stop Update
            new_stop = current_price - (2.5 * atr)
            self.stop_loss = max(self.stop_loss, new_stop) # Never lower stop
            
            if current_price < low_n:
                return {'action': 'SELL', 'reason': 'Channel Exit'}
            
            if current_price < self.stop_loss:
                return {'action': 'SELL', 'reason': 'Stop Loss'}

        return {'action': 'HOLD'}

    def get_atr(self, df, period=14):
        # Calculate True Range manually (pandas_ta unavailable)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]