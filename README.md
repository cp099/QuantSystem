# QuantSystem V2.0: Adaptive Bayesian Markov-Switching Kernel

## 1. System Philosophy
A headless, institutional-grade algorithmic trading system designed for non-stationary, adversarial markets. The system prioritizes capital preservation via a proprietary Bayesian adaptive engine.

## 2. Core Architecture
- **Brain (`src/brain/`):** Houses the ABMSM (Adaptive Bayesian Markov-Switching Model). This is the proprietary core that performs online learning and entropy-based uncertainty tracking.
- **Engine (`src/engine/`):** The operational core. Handles multi-asset data normalization (FX-adjusted), feature engineering (ATR-normalized), and risk-managed portfolio execution.
- **Strategies (`src/strategies/`):** Specialist engines (Trend, Mean Reversion, Relative Strength, Volatility Breakout) that only activate under specific Bayesian regime probabilities.
- **Research (`src/research/`):** Headless validation tools including Monte Carlo simulations and Stress-Testing modules.

## 3. Usage
- **Data Update:** `python -m src.engine.data_loader`
- **Model Init:** `python scripts/init_proprietary_model.py`
- **Headless Simulation:** `python main.py`
- **Unit Tests:** `pytest`