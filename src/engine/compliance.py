"""
Aether Bayesian Kernel - Pre-Trade Compliance Gateway
Implements SEC Rule 15c3-5 (Market Access) and SEBI algorithmic trading 
risk controls, including capital limits, concentration caps, and price band checks.
"""

class PreTradeGateway:
    """
    Coordinates pre-trade risk checks and enforces regulatory compliance.
    """

    def __init__(self, max_weight=0.30, min_equity_floor=70000.0, otr_limit=50.0):
        """
        Initializes compliance limits.

        Args:
            max_weight (float): Maximum allowed concentration for a single asset.
            min_equity_floor (float): Absolute equity floor. Breaches trigger emergency halts.
            otr_limit (float): Maximum Order-to-Trade Ratio before throttling.
        """
        self.max_weight = max_weight
        self.min_equity_floor = min_equity_floor
        self.otr_limit = otr_limit
        self.total_orders = 0
        self.total_trades = 0

    def validate_allocations(self, target_allocations, current_equity):
        """
        Applies SEC 15c3-5 Capital Limit Guard and SEBI concentration capping.

        Args:
            target_allocations (dict): Proposed ticker-to-weight dictionary.
            current_equity (float): Current portfolio valuation.

        Returns:
            dict: Regulated allocations adhering to pre-trade risk controls.
        """
        # --- SEC 15c3-5 CAPITAL PROTECTION HALT ---
        if current_equity < self.min_equity_floor:
            print(f"[COMPLIANCE GATE] EMERGENCY HALT: EQUITY ${current_equity:,.2f} BELOW FLOOR ${self.min_equity_floor:,.2f}")
            return {ticker: 0.0 for ticker in target_allocations.keys()}

        validated_allocs = {}
        total_weight = 0.0

        for ticker, weight in target_allocations.items():
            # --- SEBI CONCENTRATION GATING ---
            # Caps single asset exposure to prevent capital crowding
            capped_weight = min(max(0.0, weight), self.max_weight)
            validated_allocs[ticker] = capped_weight
            total_weight += capped_weight

        # Portfolio Leverage Normalization (max 1.0x unleveraged mandate / or custom)
        if total_weight > 1.0:
            validated_allocs = {k: v / total_weight for k, v in validated_allocs.items()}

        return validated_allocs

    def check_otr(self):
        """
        Monitors Order-to-Trade Ratio (OTR) to prevent exchange thrashing (SEBI compliance).

        Returns:
            bool: True if OTR is within compliant boundaries, False if throttled.
        """
        if self.total_trades == 0:
            return self.total_orders <= self.otr_limit
        
        otr = self.total_orders / self.total_trades
        if otr > self.otr_limit:
            print(f"[COMPLIANCE ALERT] OTR BREACH DETECTED: {otr:.1f}x. SIGNAL THROTTLED.")
            return False
        return True

    def validate_price_band(self, ticker, price, prev_close):
        """
        Validates trade execution against exchange price bands (circuit limits).

        Args:
            ticker (str): Asset identifier.
            price (float): Proposed execution price.
            prev_close (float): Previous trading session close price.

        Returns:
            bool: True if price is within dynamic bands, False if circuit limit hit.
        """
        if prev_close <= 0 or price <= 0:
            return True
        
        pct_change = abs(price - prev_close) / prev_close
        
        # Simulated 10% daily circuit limits
        if pct_change >= 0.09:  # 9% trigger to prevent hitting hard 10% limit
            print(f"[COMPLIANCE GATE] ORDER BLOCKED: {ticker} price {price:.2f} near daily circuit band (Change: {pct_change:.1%})")
            return False
        return True
