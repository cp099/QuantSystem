import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from fpdf import FPDF
from datetime import datetime

class SentinelReport:
    def __init__(self, ticker, local_ccy="INR"):
        self.ticker = ticker
        self.local_ccy = local_ccy
        self.report_dir = "reports"
        self.temp_dir = "reports/temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.navy, self.crimson = "#001F3F", "#943126"

    def _generate_visuals(self, hist_local, bench_hist, metrics):
        dates = metrics['DrawdownSeries'].index
        plt.rcParams.update({'font.size': 8, 'figure.titlesize': 10})

        # 1. Performance (Scaled down)
        plt.figure(figsize=(10, 4))
        plt.plot(dates, hist_local, color=self.navy, linewidth=1.5, label='Kernel')
        plt.plot(dates, bench_hist, color='#CCCCCC', linewidth=1, linestyle='--', label='Bench')
        plt.fill_between(dates, hist_local, 100000, color=self.navy, alpha=0.05)
        plt.title(f"Cumulative Performance Attribution: {self.ticker}", fontweight='bold')
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(f"{self.temp_dir}/p.png", dpi=200)
        plt.close()

        # 2. Drawdown (Scaled down)
        plt.figure(figsize=(10, 2))
        dd = metrics['DrawdownSeries'] * 100
        plt.fill_between(dates, dd, 0, color=self.crimson, alpha=0.2)
        plt.plot(dates, dd, color=self.crimson, linewidth=0.8)
        plt.title("Intra-Period Drawdown Profile (%)", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.temp_dir}/d.png", dpi=200)
        plt.close()

    def build_report(self, hist_local, bench_hist, m_l, m_u):
        self._generate_visuals(hist_local, bench_hist, m_l)
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_fill_color(0, 31, 63); pdf.rect(0, 0, 210, 30, 'F')
        pdf.set_text_color(255, 255, 255); pdf.set_font("Times", 'B', 20)
        pdf.set_xy(10, 8); pdf.cell(0, 10, "INVESTMENT STRATEGY AUDIT", ln=True)
        pdf.set_font("Times", '', 9); pdf.cell(0, 5, f"Ticker: {self.ticker} | Generated: {datetime.now().strftime('%Y-%m-%d')}", ln=True)

        # Metrics
        pdf.set_y(35); pdf.set_text_color(0,0,0); pdf.set_font("Times", 'B', 12)
        pdf.cell(0, 10, "I. Core Performance Metrics", ln=True)
        pdf.set_draw_color(0, 31, 63); pdf.line(10, 44, 200, 44)
        
        pdf.set_y(46); pdf.set_font("Times", 'B', 8); pdf.set_fill_color(245, 245, 245)
        w1, w2 = 50, 70
        pdf.cell(w1, 7, "METRIC", 1, 0, 'L', True)
        pdf.cell(w2, 7, f"LOCAL ({self.local_ccy})", 1, 0, 'C', True)
        pdf.cell(w2, 7, "GLOBAL (USD)", 1, 1, 'C', True)
        
        pdf.set_font("Times", '', 9)
        stats = [("Total Return", m_l['Return'], m_u['Return']),
                 ("Sharpe Ratio", str(m_l['Sharpe']), str(m_u['Sharpe'])),
                 ("Max Drawdown", m_l['MaxDD'], m_u['MaxDD']),
                 ("Ann. Volatility", m_l['Volatility'], m_u['Volatility']),
                 ("Final Liquidation", f"{self.local_ccy} {m_l['Final']:,.0f}", f"$ {m_u['Final']:,.0f}")]
        for n, l, u in stats:
            pdf.cell(w1, 6, n, 1); pdf.cell(w2, 6, l, 1, 0, 'C'); pdf.cell(w2, 6, u, 1, 1, 'C')

        # Visuals (Forced onto Page 1)
        pdf.ln(5); pdf.image(f"{self.temp_dir}/p.png", x=10, w=190)
        pdf.ln(2); pdf.image(f"{self.temp_dir}/d.png", x=10, w=190)

        # Footer
        pdf.set_y(270) # Lock to bottom of Page 1
        pdf.set_font("Times", 'I', 7); pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, "CONFIDENTIAL: Proprietary ABMSM Bayesian Adaptive Kernel Audit", align='C')

        out = f"{self.report_dir}/{self.ticker}_Audit.pdf"
        pdf.output(out); print(f">>> Audit Exported: {out}")