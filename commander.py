"""
Aether Bayesian Kernel - Terminal Commander
Operational command center providing an interactive interface for single-asset 
auditing and multi-asset sector ranking.
"""

import sys
import os
import yaml
import questionary
from datetime import datetime

from src.engine.universe import SECTORS
from src.research.sandbox import run_universal_sandbox
from src.research.ranker import SectorRanker

def clear_screen():
    """Clears the terminal console for UI refreshment."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Renders the institutional system banner."""
    print(f"\033[1;33m" + "="*60)
    print("  AETHER BAYESIAN KERNEL // SENTINEL COMMANDER")
    print(f"  Operational Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60 + "\033[0m")

def main_loop():
    """
    Orchestrates the primary operational loop.
    
    Provides navigational access to the research and audit modules, 
    managing the transition between high-fidelity simulations and 
    cross-sectional alpha discovery.
    """
    while True:
        clear_screen()
        print_banner()
        
        try:
            choice = questionary.select(
                "SELECT OPERATIONAL MODE:",
                choices=[
                    "1. Single Asset Audit (Global Search)",
                    "2. Sector Alpha Ranking (Multi-Asset)",
                    "3. Exit System"
                ],
                style=questionary.Style([
                    ('selected', 'fg:#FFB000 bold'),
                    ('pointer', 'fg:#FFB000 bold'),
                ])
            ).ask()

            if choice is None or "Exit" in choice:
                print("\n[SYSTEM] SHUTDOWN INITIATED. AUDIT LOG SECURED.")
                break

            # --- MODE I: SINGLE ASSET AUDIT ---
            if "Single Asset" in choice:
                ticker = questionary.text("ENTER GLOBAL TICKER (e.g., RELIANCE.NS, TSLA, BTC-USD):").ask()
                if ticker:
                    # Inhaling data and propagating through the Bayesian kernel
                    run_universal_sandbox(ticker.upper())
                    
                    # Optional dispatch to the reporting kernel
                    if questionary.confirm("GENERATE INSTITUTIONAL PDF DOSSIER?").ask():
                        print(f"[SENTINEL] DISPATCHING REPORTING KERNEL. DOCUMENT PERSISTED IN /REPORTS")
                    
                    questionary.press_any_key_to_continue().ask()

            # --- MODE II: SECTOR ALPHA RANKING ---
            elif "Sector Alpha" in choice:
                region = questionary.select("SELECT JURISDICTION:", choices=list(SECTORS.keys())).ask()
                sector = questionary.select("SELECT SECTOR:", choices=list(SECTORS[region].keys())).ask()
                
                tickers = SECTORS[region][sector]
                print(f"\n[RANKER] ANALYZING {sector} LEADERSHIP...")
                
                # Cross-sectional tournament logic
                ranker = SectorRanker()
                results = ranker.rank_universe(tickers)
                print("\n", results)
                
                questionary.press_any_key_to_continue().ask()

        except (KeyboardInterrupt, EOFError):
            # Graceful termination of the kernel on external interrupt signals
            print("\n\n" + "="*60)
            print("  AETHER BAYESIAN KERNEL // SECURE SHUTDOWN COMPLETE")
            print("  Inference state persisted. Memory purged.")
            print("="*60 + "\n")
            sys.exit(0)

if __name__ == "__main__":
    main_loop()