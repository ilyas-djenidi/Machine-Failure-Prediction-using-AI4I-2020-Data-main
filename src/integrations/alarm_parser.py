"""
Alarm Parser for Siemens WinCC V16
Converts WinCC Alarm CSV exports into Ground Truth labels for ML retraining.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def parse_wincc_alarms(csv_path: str, output_path: str):
    """
    Parses exported WinCC Alarm logs to extract failure events (ALMTD, CRASD, CRPRD).
    Assigns binary label=1 to corresponding motor instances.
    """
    if not Path(csv_path).exists():
        logging.error(f"File {csv_path} not found.")
        return

    # Assuming standard WinCC export format (CSV with delimiter, often semicolon)
    try:
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    except Exception as e:
        df = pd.read_csv(csv_path) # Fallback to comma

    # Ensure required columns exist
    required_cols = ['DateTime', 'Instance', 'CLASS', 'TxtCame']
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Missing required column: {col}")
            return
            
    # Filter for fault classes
    fault_classes = ['ALMTD', 'CRASD', 'CRPRD']
    faults_df = df[df['CLASS'].isin(fault_classes)].copy()
    
    if faults_df.empty:
        logging.info("No fault records found in the provided alarm log.")
        return
        
    # Extract clean labels
    faults_df['Failure_Label'] = 1
    faults_df['DateTime'] = pd.to_datetime(faults_df['DateTime'], errors='coerce')
    
    # Sort and clean
    faults_df = faults_df.sort_values('DateTime')
    labels_df = faults_df[['DateTime', 'Instance', 'CLASS', 'TxtCame', 'Failure_Label']]
    
    # Save extracted labels
    labels_df.to_csv(output_path, index=False)
    logging.info(f"Extracted {len(labels_df)} fault labels. Saved to {output_path}")
    
    return labels_df

if __name__ == "__main__":
    # Example usage
    # parse_wincc_alarms("data/raw/wincc_alarms_export.csv", "data/processed/fault_labels.csv")
    print("Alarm parser ready.")
