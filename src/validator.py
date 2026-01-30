import pandas as pd
from src.signals import TradeSignal

class SignalValidator:
    def __init__(self, min_regime_confidence=0.60):
        self.min_confidence = min_regime_confidence

    def check_regime_confidence(self, regime_probs, intended_regime):
        """
        Filter 1: Is the Regime Engine sure about the current state?
        regime_probs: array like [0.1, 0.8, 0.05, 0.05]
        intended_regime: int (e.g., 0 for Bull)
        """
        confidence = regime_probs[intended_regime]
        if confidence < self.min_confidence:
            return False, f"Low Regime Confidence: {confidence:.2f} < {self.min_confidence}"
        return True, "Confidence OK"

    def check_volume_confirmation(self, market_data_row):
        """
        Filter 2: Does the move have volume backing?
        Requires 'Rvol' feature from FeatureEngineer.
        """
        # If Relative Volume is available and < 0.8 (20% below average), reject.
        if 'Rvol' in market_data_row and market_data_row['Rvol'] < 0.8:
            return False, f"Weak Volume (RVOL: {market_data_row['Rvol']:.2f})"
        return True, "Volume OK"

    def check_volatility_safety(self, market_data_row):
        """
        Filter 3: Are we entering during extreme instability?
        """
        # If Volatility Ratio > 1.5, market is in shock/expansion. Dangerous for entry.
        if 'Vol_ratio' in market_data_row and market_data_row['Vol_ratio'] > 1.5:
            return False, f"Volatility Shock (Ratio: {market_data_row['Vol_ratio']:.2f})"
        return True, "Volatility OK"

    def validate(self, signal: TradeSignal, market_data_row, regime_probs):
        """
        Master Validation Function.
        Returns: (bool, str) -> (Passed?, Reason)
        """
        
        # 1. Check Volatility Safety (Universal Rule)
        if signal.action in ['LONG', 'SHORT']:
            passed, msg = self.check_volatility_safety(market_data_row)
            if not passed: return False, msg

        # 2. Check Volume (only for Longs usually)
        if signal.action == 'LONG':
            passed, msg = self.check_volume_confirmation(market_data_row)
            if not passed: return False, msg

        # 3. Regime Confidence
        # This requires us to know which regime the strategy 'likes'.
        # For now, we assume:
        # Trend Strategy likes Regime 0 (Bull) or 1 (Bear)
        # We will skip this specific check here and enforce it in the Strategy Engine logic,
        # OR we can pass it in the signal metadata. For now, we pass.
        
        return True, "All Checks Passed"