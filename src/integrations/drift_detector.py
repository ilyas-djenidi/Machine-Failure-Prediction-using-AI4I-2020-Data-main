"""
Data Drift Detector
Compares live factory sensor data distributions against the AI4I 2020 training distributions.
Uses Kolmogorov-Smirnov (KS) tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

class DriftDetector:
    def __init__(self, reference_csv: str):
        self.ref_csv = reference_csv
        self.ref_df = None
        self.features = [
            "Air temperature [K]", "Process temperature [K]", 
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
        ]
        
    def load_reference(self):
        try:
            self.ref_df = pd.read_csv(self.ref_csv)
            # Ensure columns exist
            for f in self.features:
                if f not in self.ref_df.columns:
                    logging.warning(f"Feature {f} not found in reference data.")
            logging.info(f"Loaded reference data with {len(self.ref_df)} samples.")
        except Exception as e:
            logging.error(f"Failed to load reference data: {e}")
            
    def detect_drift(self, current_data: pd.DataFrame, p_value_threshold: float = 0.05) -> dict:
        """
        Runs KS-test on current data vs reference data.
        Returns drift analysis dict.
        """
        if self.ref_df is None:
            self.load_reference()
            if self.ref_df is None:
                return {"error": "Reference data not available."}
                
        results = {"drift_detected": False, "features": {}}
        drift_count = 0
        
        for feature in self.features:
            if feature in current_data.columns and feature in self.ref_df.columns:
                ref_vals = self.ref_df[feature].dropna().values
                curr_vals = current_data[feature].dropna().values
                
                if len(curr_vals) < 30:
                    results["features"][feature] = {"status": "insufficient_data"}
                    continue
                    
                # KS test
                stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
                
                is_drifting = bool(p_val < p_value_threshold)
                if is_drifting:
                    drift_count += 1
                    
                results["features"][feature] = {
                    "p_value": round(float(p_val), 5),
                    "ks_stat": round(float(stat), 5),
                    "is_drifting": is_drifting
                }
                
        # Alert if 3+ features drift
        if drift_count >= 3:
            results["drift_detected"] = True
            results["message"] = "CRITICAL: Widespread feature drift detected. Consider retraining."
        elif drift_count > 0:
            results["message"] = "WARNING: Partial feature drift detected."
        else:
            results["message"] = "OK: Data distributions align with training data."
            
        return results

if __name__ == "__main__":
    # Example usage
    # detector = DriftDetector("data/ai4i2020.csv")
    # live_data = pd.read_csv("data/live_factory_export.csv")
    # report = detector.detect_drift(live_data)
    # print(json.dumps(report, indent=2))
    print("Drift detector ready.")
