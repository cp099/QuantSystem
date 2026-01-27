from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    def __init__(self, name):
        self.name = name
        self.position = 0  # 1 (Long), -1 (Short), 0 (Flat)
        self.entry_price = 0.0
        self.stop_loss = 0.0
        
    @abstractmethod
    def generate_signal(self, market_data, regime_probs):
        """
        Input: 
            market_data (pd.DataFrame): Recent price history
            regime_probs (list): [P(Steady), P(Crash), P(VolBull), P(Chop)]
        Output:
            dict: {'action': 'BUY'/'SELL'/'HOLD', 'confidence': 0.0-1.0}
        """
        pass
    
    def calculate_volatility(self, prices, window=20):
        # Annualized Volatility
        log_ret = np.log(prices / prices.shift(1))
        return log_ret.rolling(window).std().iloc[-1] * np.sqrt(252)

    def check_invalidation(self, current_price):
        # Standard exit logic
        if self.position == 1 and current_price < self.stop_loss:
            return True
        return False