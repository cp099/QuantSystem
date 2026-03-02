"""
Aether Bayesian Kernel - Investment Mandate & Universe Definition
Static configuration of global market segments, sector-specific baskets, 
and institutional liquidity proxies.
"""

# Dictionary defining the cross-sectional scope of the system
SECTORS = {
    "JURISDICTION: INDIA (NSE)": {
        "FINANCIAL_SERVICES": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
        "TECHNOLOGY_SERVICES": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        "AUTOMOTIVE_MANUFACTURING": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS"],
        "CORE_EQUITIES": ["RELIANCE.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS"]
    },
    "JURISDICTION: USA (NYSE/NASDAQ)": {
        "TECHNOLOGY_LEADERS": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "FINANCIAL_INSTITUTIONS": ["JPM", "BAC", "GS", "MS", "V"],
        "ENERGY_INFRASTRUCTURE": ["XOM", "CVX", "SLB", "COP"],
        "SEMICONDUCTOR_EQUITY": ["AMD", "INTC", "TSM", "AVGO", "ASML"]
    },
    "GLOBAL_MACRO_PROXIES": {
        "COMMODITY_COMPLEX": ["GLD", "SLV", "USO", "UNG", "CPER"],
        "FOREIGN_EXCHANGE": ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDINR=X"],
        "EQUITY_INDICES": ["SPY", "QQQ", "IWM", "^NSEI", "^GDAXI"]
    }
}