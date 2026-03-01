import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class SentinelReport:
    """Generates Institutional Visual Tear Sheets (Headless)."""
    def __init__(self, ticker):
        self.ticker = ticker
        plt.style.use('dark_background')

    def generate(self, hist_local, hist_usd, bench_hist, local_ccy):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2)
        
        # 1. MAIN PERFORMANCE CHART
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.plot(hist_local, color='#00FF00', label=f'ABMSM Kernel ({local_ccy})', linewidth=2)
        ax1.plot(bench_hist, color='#AAAAAA', label='Benchmark (Index)', linestyle='--', alpha=0.7)
        ax1.set_title(f"INSTITUTIONAL PERFORMANCE AUDIT: {self.ticker}", color='#FFB000', fontsize=16)
        ax1.legend()
        ax1.grid(alpha=0.1)

        # 2. DRAWDOWN CHART
        ax2 = fig.add_subplot(gs[2, 0])
        dd = (pd.Series(hist_local) / pd.Series(hist_local).cummax() - 1)
        ax2.fill_between(range(len(dd)), dd, color='#FF3333', alpha=0.3)
        ax2.plot(dd, color='#FF3333', linewidth=1)
        ax2.set_title("Drawdown Profile", color='#FF3333')

        # 3. CURRENCY IMPACT CHART
        ax3 = fig.add_subplot(gs[2, 1])
        impact = (pd.Series(hist_usd) / pd.Series(hist_usd).iloc[0]) / (pd.Series(hist_local) / pd.Series(hist_local).iloc[0])
        ax3.plot(impact, color='#00A6A9', label='FX Impact (USD/Local)')
        ax3.set_title("Currency Reflexivity Sense", color='#00A6A9')
        ax3.fill_between(range(len(impact)), 1, impact, where=(impact < 1), color='red', alpha=0.2)
        
        plt.tight_layout()
        os.makedirs("reports", exist_ok=True)
        report_path = f"reports/{self.ticker.replace('.','_')}_tearsheet.png"
        plt.savefig(report_path)
        print(f">>> SENTINEL: Visual Tear Sheet generated at {report_path}")
        plt.close()