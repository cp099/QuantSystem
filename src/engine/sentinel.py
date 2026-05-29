"""
Aether Bayesian Kernel - Decision Audit Sentinel
Coordinates the persistence of recursive inference states and implements 
proactive risk monitoring via entropy thresholds.
"""

import json
import os
from datetime import datetime

class Sentinel:
    """
    Coordinates state-persistence and risk-alerting protocols.
    
    Maintains an immutable record of Bayesian beliefs, uncertainty metrics, 
    and portfolio composition to provide full auditability of the kernel logic.
    """

    def __init__(self, log_dir="logs/audit"):
        """
        Initializes the audit persistence directory.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_decision(self, date, regime_probs, entropy, holdings, equity=0.0):
        """
        Persists a high-fidelity snapshot of the kernel state to disk.
        
        Args:
            date (datetime): Current simulation timestamp.
            regime_probs (np.ndarray): Vector of state probabilities.
            entropy (float): Normalized Shannon Entropy of the belief.
            holdings (dict): Current asset-level unit distribution.
            equity (float): Current total portfolio equity valuation.
        """
        # Structured audit schema for forensic performance analysis
        audit_entry = {
            "timestamp": str(date),
            "dominant_regime": int(regime_probs.argmax()),
            "regime_confidence": float(regime_probs.max()),
            "entropy": float(entropy),
            "holdings": {k: float(v) for k, v in holdings.items()},
            "equity": float(equity)
        }
        
        filename = f"{self.log_dir}/decision_audit.jsonl"
        with open(filename, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

    def generate_risk_alert(self, entropy):
        """
        Evaluates kernel uncertainty for proactive risk mitigation.
        
        Monitors belief entropy; values exceeding the 0.8 threshold trigger 
        systemic risk alerts to indicate high-regime ambiguity.
        """
        if entropy > 0.8:
            print(f"[SENTINEL RISK ALERT] CRITICAL ENTROPY DETECTED: {entropy:.2f} (REGIME AMBIGUITY)")