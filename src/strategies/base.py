"""
Aether Bayesian Kernel - Strategy Abstraction Layer
Defines the standard interface for specialist strategy engines, 
facilitating modular signal generation across global asset universes.
"""

from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Abstract base class for all proprietary strategy specialists.
    
    Provides a standardized contract for translating relativistic market data 
    and Bayesian state beliefs into asset-level directional convictions.
    """

    def __init__(self, name):
        """
        Initializes the strategy engine.
        
        Args:
            name (str): Unique identifier for the strategy engine.
        """
        self.name = name
        
    @abstractmethod
    def generate_signals(self, market_data_slice, regime_probs):
        """
        Calculates directional conviction weights for a cross-sectional market slice.
        
        Args:
            market_data_slice (dict): Collection of standardized data rows for 
                                      all assets in the current period.
            regime_probs (np.ndarray): Vector of current Bayesian state probabilities.
            
        Returns:
            dict: Mapping of ticker identifiers to conviction weights [0.0 - 1.0].
        """
        pass