# Predictive Maintenance System - Dataset Information

This directory contains all datasets used for the predictive maintenance system.

## Datasets Overview

### 1. UCI AI4I 2020 Predictive Maintenance Dataset
- **Source**: UCI Machine Learning Repository (ID: 601)
- **Description**: Synthetic dataset reflecting real predictive maintenance data
- **Features**: Air temperature, process temperature, rotational speed, torque, tool wear
- **Target**: Machine failure (binary) + failure type categories
- **Samples**: ~10,000
- **Use Case**: Primary dataset for training classification models

### 2. Machine Failure Dataset (Kaggle)
- **Source**: Kaggle - kartikaytandon/machine-failure
- **Description**: Industrial machine failure data with sensor readings
- **Features**: Various sensor measurements and operational parameters
- **Target**: Failure indicators
- **Use Case**: Additional training data and cross-validation

### 3. NASA CMAPS Turbofan Degradation
- **Source**: Kaggle - behrad3d/nasa-cmaps
- **Description**: Turbofan engine degradation simulation data
- **Features**: 21 sensors + 3 operational settings
- **Format**: Time-series data with run-to-failure trajectories
- **Samples**: Multiple engines with varying lifespans
- **Use Case**: Time-series LSTM training and degradation pattern analysis

### 4. Microsoft Azure Predictive Maintenance
- **Source**: Kaggle - arnabbiswas1/microsoft-azure-predictive-maintenance
- **Description**: Multi-table dataset with telemetry, errors, maintenance, and machine info
- **Tables**:
  - Telemetry: Time-series sensor data
  - Errors: Error logs
  - Maintenance: Maintenance records
  - Machines: Machine metadata
  - Failures: Failure records
- **Use Case**: Complex multi-table analysis and realistic industrial scenario

## Directory Structure

```
data/
├── raw/                    # Downloaded datasets (original format)
│   ├── uci_ai4i/
│   ├── machine_failure.csv
│   ├── nasa_cmaps/
│   └── azure_maintenance/
├── processed/              # Preprocessed and cleaned data
│   ├── train/
│   ├── val/
│   └── test/
└── datasets_info.md       # This file
```

## Data Loading

Use the `DatasetLoader` class from `src/data/data_loader.py`:

```python
from src.data.data_loader import DatasetLoader

loader = DatasetLoader()
all_datasets = loader.load_all_datasets()
```

## Citation & Attribution

- UCI AI4I: UCI Machine Learning Repository
- NASA CMAPS: NASA Ames Prognostics Data Repository
- Azure: Microsoft Azure AI Gallery
- Machine Failure: Kaggle Community Dataset
