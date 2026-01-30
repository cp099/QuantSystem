from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeSignal:
    """
    Standardized message format from Strategy Engines to the Execution System.
    """
    symbol: str
    action: str           # 'LONG', 'SHORT', 'FLAT'
    strategy_name: str
    timestamp: datetime
    
    # Context data for the Validator
    strength: float = 1.0 # 0.0 to 1.0 (Confidence)
    comment: str = ""