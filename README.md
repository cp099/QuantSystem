# AETHER BAYESIAN KERNEL (ABK) // SENTINEL OS
**Institutional-Grade Adaptive Quantitative Trading Operating System**

![Kernel Version](https://img.shields.io/badge/Kernel-V3.5.0_Stable-003366)
![IP Status](https://img.shields.io/badge/IP_Status-Proprietary_&_Protected-C5A059)
![Environment](https://img.shields.io/badge/Environment-Headless_Quant_Kernel-black)

## I. Mission Profile
The Aether Bayesian Kernel (ABK) is a proprietary computational framework designed for high-stakes navigation of non-stationary, adversarial global markets. Developed as a headless operational kernel, ABK utilizes recursive Bayesian inference to mitigate systemic tail-risk and harvest alpha in volatile regimes.

## II. Core Mathematical Differentiation
Unlike static statistical models, ABK utilizes a proprietary **Adaptive Bayesian Markov-Switching Model (ABMSM)**.
*   **Online Learning:** Recursive parameter adaptation utilizing Dirichlet-Multinomial transition memory.
*   **7-Dimensional Perception:** Standardized state-space tracking Velocity, Risk, Compression, Alpha, Vol-Divergence, Liquidity, and Breadth.
*   **Scale-Invariance:** Relativistic robust scaling (IQR-based) allows for uniform signal generation across divergent jurisdictions (e.g., NSE, NYSE, Crypto).

## III. System Architecture
- **`src/brain/`**: **[PROTECTED CORE]** Implements the ABMSM proprietary math and entropy-based uncertainty filters.
- **`src/engine/`**: Operational infrastructure including universal data loaders, currency-basis synchronization, and institutional risk management.
- **`src/research/`**: Validation laboratory containing Monte Carlo path-randomization, cross-sectional ranking, and synthetic stress-test engines.
- **`scripts/`**: Protocols for kernel initialization and global instinct training.
- **`templates/`**: HTML/JS user interface mockups designed to resemble an institutional-grade financial terminal.
- **`dashboard_server.py`**: Lightweight HTTP backend server enabling historical backtests, real-time paper trading loops, and MLP network diagram visualizations.

## IV. Interactive Financial Terminal UI & Features
The system includes an interactive, professional financial terminal UI (`http://localhost:8080`) providing:
1.  **Command Prompt Console Bar**: Type tickers and execution functions (e.g., `AAPL GP`, `ETERNAL.NS BT`, `ALLOC 150000`, `MONITOR`, `HELP`). Supports real-time ticker suggestions and arrow-key autocomplete dropdown navigation.
2.  **ZOMATO Rebrand Redirection**: Automatically resolves old delisted Zomato symbols (`ZOMATO.NS`, `ZOMATO.BO`, `ZOMATO`) to their active rebranded counterparts (`ETERNAL.NS` / `ETERNAL.BO`, effective April 2025) and alerts the user in the command console feed.
3.  **Lightweight Caching & Live-Patching**: Implements a 5-minute historical base cache and only requests the latest single real-time bar (`yf.Ticker.history(period="1d")`, <100ms) for target tickers on subsequent polls. Merges live ticks with cached history in-memory to prevent redundant yfinance downloads.
4.  **1D / 5D Granular Timeframe Charts**: Timeframe selectors include `1D` and `5D` options, pulling high-density intraday data (`5m` intervals for 1D, `15m` intervals for 5D) to render detailed line charts.
5.  **Merged Sliding Live Charts**: The live paper trading visualizer combines historical pre-loaded bars with real-time incoming ticks. Scales and scrolls smoothly from right to left using a moving window (80 to 500 points depending on the timeframe) to prevent graph compression.
6.  **Dual Currency Support**: Formatting values dynamically transitions to Indian Rupees (`₹`) or United States Dollars (`$`) depending on the suffix of the target asset (`.NS` or `.BO` format).
7.  **Browser Wall Clock**: Status bar clock updates dynamically every second using browser time.

## V. Operational Protocol
To execute an audit or initialize the kernel, utilize the following command structures:

1.  **Environment Synchronization:**
    `pip install -r requirements.txt`
2.  **Kernel Initialization:**
    `python scripts/train_instinct.py`
3.  **Terminal Dashboard Launch:**
    `python dashboard_server.py`
    *Navigate to `http://localhost:8080` in your web browser.*
4.  **Command Center Launch (Headless CLI):**
    `python commander.py`
5.  **Unit Testing Audit:**
    `pytest`

## VI. Intellectual Property & Licensing
The mathematical algorithms and source code contained within the `src/brain` directory are the exclusive **Intellectual Property** of Chirag P Patil. This repository is provided for institutional audit, validation, and proof-of-concept purposes only. 

**Unauthorized distribution or reverse engineering is strictly prohibited.**

For licensing inquiries or technical documentation: **chiragpatil07@gmail.com**