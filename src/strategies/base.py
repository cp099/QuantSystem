from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Strategy(ABC):
    def __init__(self, name):
        self.name = name
        self.position = 0 # 0 = Flat, 1 = Long, -1 = Short
        self.equity_curve = []
        
    @abstractmethod
    def generate_signal(self, current_bar, current_regime):
        """
        Input:
        - current_bar: A dictionary/series with 'Close', 'Open', 'High', 'Low'
        - current_regime: Integer (0, 1, 2, 3)
        
        Output:
        - Signal: 1.0 (Buy), -1.0 (Sell), 0.0 (Hold)
        """
        pass