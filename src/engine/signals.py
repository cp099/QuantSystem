"""
Aether Bayesian Kernel - Signal Encapsulation Protocol
Defines the standardized data structure for cross-modular communication of 
directional convictions between strategy engines and the portfolio kernel.
"""

from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeSignal:
    """
    Agnostic container for directional convictions.
    
    Standardizes the metadata required for the risk engine to perform 
    capital allocation based on specific strategy outputs.
    
    Attributes:
        symbol (str): Global security identifier.
        action (str): Directional intent (e.g., 'LONG', 'SHORT', 'FLAT').
        strategy_name (str): Identifier for the originating strategy engine.
        timestamp (datetime): Temporal anchor for the signal generation.
        strength (float): Conviction magnitude utilized for risk weighting [0.0 - 1.0].
        comment (str): Metadata for audit persistence.
    """
    symbol: str
    action: str           
    strategy_name: str
    timestamp: datetime
    strength: float = 1.0 
    comment: str = ""