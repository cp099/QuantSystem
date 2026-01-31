import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def generate_signals(self, market_data_slice, regime_probs):
        """
        V2 Interface:
        Input:
        - market_data_slice: A dictionary { 'SPY': row_series, 'GLD': row_series, ... }
                             containing the data for the current timestamp for all assets.
        - regime_probs: Array of probabilities [P(Regime0), P(Regime1), ...] from ABMSM.
        
        Output:
        - signals: A dictionary { 'SPY': 1.0, 'GLD': 0.0, ... }
                   representing target allocation weights (0.0 to 1.0) for this specific engine.
        """
        pass