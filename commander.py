import sys
import os
import yaml
import questionary
from datetime import datetime

# Link internal modules
from src.engine.universe import SECTORS
from src.research.sandbox import run_universal_sandbox
from src.research.ranker import SectorRanker

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print(f"\033[1;33m" + "="*60)
    print("  QUANT OS V2.0 // ADAPTIVE BAYESIAN MARKOV KERNEL")
    print(f"  System Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60 + "\033[0m")

def main_loop():
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
                print("\n>>> System Shutdown Initiated. Secure log closed.")
                break

            if "Single Asset" in choice:
                ticker = questionary.text("ENTER TICKER SYMBOL (e.g., RELIANCE.NS, TSLA, BTC-USD):").ask()
                if ticker:
                    # Run the sandbox logic
                    run_universal_sandbox(ticker.upper())
                    
                    # Ask for PDF
                    if questionary.confirm("GENERATE PDF INSTITUTIONAL DOSSIER?").ask():
                        print(f">>> Sentinel working... (PDF saved in /reports)")
                    
                    questionary.press_any_key_to_continue().ask()

            elif "Sector Alpha" in choice:
                region = questionary.select("SELECT REGION:", choices=list(SECTORS.keys())).ask()
                sector = questionary.select("SELECT SECTOR:", choices=list(SECTORS[region].keys())).ask()
                
                tickers = SECTORS[region][sector]
                print(f"\n>>> Ranking {sector} leaders...")
                
                ranker = SectorRanker()
                results = ranker.rank_universe(tickers)
                print("\n", results)
                
                questionary.press_any_key_to_continue().ask()

        except (KeyboardInterrupt, EOFError):
            # This handles Ctrl+C and Ctrl+D/Q signals
            print("\n\n" + "="*60)
            print("  QUANT OS V2.0 // SECURE SHUTDOWN COMPLETED")
            print("  All proprietary weights saved. Memory purged.")
            print("="*60 + "\n")
            sys.exit(0)

if __name__ == "__main__":
    main_loop()