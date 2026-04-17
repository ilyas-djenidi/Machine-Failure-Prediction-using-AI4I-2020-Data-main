"""
Dataset Download Script
Downloads all 4 datasets using the user-provided code
"""

print("="*60)
print("Downloading Predictive Maintenance Datasets")
print("="*60)

# 1. Install required packages
print("\n[1/4] Installing ucimlrepo...")
import subprocess
import sys

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "ucimlrepo"])
    print("✓ ucimlrepo installed")
except Exception as e:
    print(f"Note: {e}")

# 2. Download UCI AI4I 2020 Dataset
print("\n[2/4] Downloading UCI AI4I 2020 Dataset...")
try:
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601) 
    
    # data (as pandas dataframes) 
    X = ai4i_2020_predictive_maintenance_dataset.data.features 
    y = ai4i_2020_predictive_maintenance_dataset.data.targets 
    
    # metadata 
    print("Metadata:")
    print(ai4i_2020_predictive_maintenance_dataset.metadata) 
    
    # variable information 
    print("\nVariable Information:")
    print(ai4i_2020_predictive_maintenance_dataset.variables)
    
    # Save to disk
    import os
    os.makedirs("data/raw/uci_ai4i", exist_ok=True)
    X.to_csv("data/raw/uci_ai4i/features.csv", index=False)
    y.to_csv("data/raw/uci_ai4i/targets.csv", index=False)
    print(f"✓ UCI AI4I dataset saved: {X.shape[0]} samples, {X.shape[1]} features")
    
except Exception as e:
    print(f"✗ Error downloading UCI AI4I: {e}")

# 3. Install kagglehub
print("\n[3/4] Installing kagglehub...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "kagglehub"])
    print("✓ kagglehub installed")
except Exception as e:
    print(f"Note: {e}")

# 4. Download Kaggle datasets
print("\n[4/4] Downloading Kaggle datasets...")
try:
    import kagglehub
    
    # Azure dataset
    print("\n  [a] Microsoft Azure Predictive Maintenance...")
    path1 = kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")
    print(f"  ✓ Path to dataset files: {path1}")
    
    # NASA CMAPS
    print("\n  [b] NASA CMAPS...")
    path2 = kagglehub.dataset_download("behrad3d/nasa-cmaps")
    print(f"  ✓ Path to dataset files: {path2}")
    
    # Machine Failure
    print("\n  [c] Machine Failure...")
    path3 = kagglehub.dataset_download("kartikaytandon/machine-failure")
    print(f"  ✓ Path to dataset files: {path3}")
    
    # Save paths to a file for reference
    with open("data/raw/dataset_paths.txt", "w") as f:
        f.write(f"Azure Maintenance: {path1}\n")
        f.write(f"NASA CMAPS: {path2}\n")
        f.write(f"Machine Failure: {path3}\n")
    
    print("\n✓ All datasets downloaded successfully!")
    print(f"\nDataset locations saved to: data/raw/dataset_paths.txt")
    
except Exception as e:
    print(f"✗ Error downloading Kaggle datasets: {e}")
    print("\nNote: Make sure you have kaggle API credentials configured.")
    print("See: https://www.kaggle.com/docs/api")

print("\n" + "="*60)
print("Download Complete!")
print("="*60)
