import json
import os
from datetime import datetime

class Sentinel:
    def __init__(self, log_dir="logs/audit"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_decision(self, date, regime_probs, entropy, holdings):
        """Saves a permanent record of the Brain's state."""
        audit_entry = {
            "timestamp": str(date),
            "dominant_regime": int(regime_probs.argmax()),
            "regime_confidence": float(regime_probs.max()),
            "entropy": float(entropy),
            "holdings": {k: float(v) for k, v in holdings.items()}
        }
        
        filename = f"{self.log_dir}/decision_audit.jsonl"
        with open(filename, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

    def generate_risk_alert(self, entropy):
        """Alerts if the model is confused (Proprietary Sense)."""
        if entropy > 0.8:
            print(f"!!! SENTINEL RISK ALERT: Model Entropy at {entropy:.2f} (Extreme Uncertainty) !!!")