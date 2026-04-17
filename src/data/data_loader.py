"""
Data Loader Module for Predictive Maintenance System
Downloads and loads all 4 datasets with standardized format
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles downloading and loading of all predictive maintenance datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the DatasetLoader
        
        Args:
            data_dir: Directory where raw datasets will be stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_uci_ai4i(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load UCI AI4I 2020 Predictive Maintenance Dataset
        
        Returns:
            Tuple of (features DataFrame, targets DataFrame)
        """
        logger.info("Loading UCI AI4I 2020 dataset...")
        
        try:
            from ucimlrepo import fetch_ucirepo
            
            # Fetch dataset
            dataset = fetch_ucirepo(id=601)
            
            # Get features and targets
            X = dataset.data.features
            y = dataset.data.targets
            
            # Save to disk for caching
            cache_path = self.data_dir / "uci_ai4i"
            cache_path.mkdir(exist_ok=True)
            X.to_csv(cache_path / "features.csv", index=False)
            y.to_csv(cache_path / "targets.csv", index=False)
            
            logger.info(f"✓ UCI AI4I dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"  Metadata: {dataset.metadata}")
            logger.info(f"  Variables: {dataset.variables}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading UCI AI4I dataset: {e}")
            # Try to load from cache
            cache_path = self.data_dir / "uci_ai4i"
            if (cache_path / "features.csv").exists():
                logger.info("Loading from cache...")
                X = pd.read_csv(cache_path / "features.csv")
                y = pd.read_csv(cache_path / "targets.csv")
                return X, y
            raise
    
    def load_machine_failure(self) -> pd.DataFrame:
        """
        Load Machine Failure dataset from Kaggle
        
        Returns:
            DataFrame with all data
        """
        logger.info("Loading Machine Failure dataset...")
        
        try:
            import kagglehub
            
            # Download dataset
            path = kagglehub.dataset_download("kartikaytandon/machine-failure")
            logger.info(f"Dataset downloaded to: {path}")
            
            # Find CSV file
            csv_files = list(Path(path).rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in downloaded dataset")
            
            # Load the data
            df = pd.read_csv(csv_files[0])
            
            # Cache it
            cache_path = self.data_dir / "machine_failure.csv"
            df.to_csv(cache_path, index=False)
            
            logger.info(f"✓ Machine Failure dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            logger.info(f"  Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Machine Failure dataset: {e}")
            # Try cache
            cache_path = self.data_dir / "machine_failure.csv"
            if cache_path.exists():
                logger.info("Loading from cache...")
                return pd.read_csv(cache_path)
            raise
    
    def load_nasa_cmaps(self) -> Dict[str, pd.DataFrame]:
        """
        Load NASA CMAPS Turbofan Degradation dataset
        
        Returns:
            Dictionary containing train and test DataFrames
        """
        logger.info("Loading NASA CMAPS dataset...")
        
        try:
            import kagglehub
            
            # Download dataset
            path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
            logger.info(f"Dataset downloaded to: {path}")
            
            # Find all files
            data_files = list(Path(path).rglob("*.txt")) + list(Path(path).rglob("*.csv"))
            
            # NASA CMAPS has multiple train/test sets
            datasets = {}
            for file in data_files:
                df = self._load_nasa_file(file)
                if df is not None:
                    datasets[file.stem] = df
            
            # Cache
            cache_dir = self.data_dir / "nasa_cmaps"
            cache_dir.mkdir(exist_ok=True)
            for name, df in datasets.items():
                df.to_csv(cache_dir / f"{name}.csv", index=False)
            
            logger.info(f"✓ NASA CMAPS dataset loaded: {len(datasets)} sub-datasets")
            for name, df in datasets.items():
                logger.info(f"  {name}: {df.shape[0]} samples, {df.shape[1]} features")
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading NASA CMAPS dataset: {e}")
            # Try cache
            cache_dir = self.data_dir / "nasa_cmaps"
            if cache_dir.exists():
                logger.info("Loading from cache...")
                datasets = {}
                for file in cache_dir.glob("*.csv"):
                    datasets[file.stem] = pd.read_csv(file)
                return datasets
            raise
    
    def _load_nasa_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Helper to load NASA CMAPS files which have specific format"""
        try:
            # NASA CMAPS files are space-separated without headers
            # Columns: unit_id, time_cycles, operational_setting_1-3, sensor_1-21
            col_names = ['unit_id', 'time_cycles'] + \
                       [f'op_setting_{i}' for i in range(1, 4)] + \
                       [f'sensor_{i}' for i in range(1, 22)]
            
            if filepath.suffix == '.txt':
                df = pd.read_csv(filepath, sep=' ', header=None, names=col_names[:100])
                # Remove extra columns if any
                df = df.dropna(axis=1, how='all')
            else:
                df = pd.read_csv(filepath)
            
            return df
        except Exception as e:
            logger.warning(f"Could not load {filepath}: {e}")
            return None
    
    def load_azure_maintenance(self) -> Dict[str, pd.DataFrame]:
        """
        Load Microsoft Azure Predictive Maintenance dataset
        
        Returns:
            Dictionary containing different data tables (telemetry, errors, maintenance, etc.)
        """
        logger.info("Loading Azure Predictive Maintenance dataset...")
        
        try:
            import kagglehub
            
            # Download dataset
            path = kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")
            logger.info(f"Dataset downloaded to: {path}")
            
            # Azure dataset has multiple CSV files
            csv_files = list(Path(path).rglob("*.csv"))
            
            datasets = {}
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    datasets[file.stem] = df
                    logger.info(f"  Loaded {file.stem}: {df.shape}")
                except Exception as e:
                    logger.warning(f"Could not load {file}: {e}")
            
            # Cache
            cache_dir = self.data_dir / "azure_maintenance"
            cache_dir.mkdir(exist_ok=True)
            for name, df in datasets.items():
                df.to_csv(cache_dir / f"{name}.csv", index=False)
            
            logger.info(f"✓ Azure dataset loaded: {len(datasets)} tables")
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading Azure dataset: {e}")
            # Try cache
            cache_dir = self.data_dir / "azure_maintenance"
            if cache_dir.exists():
                logger.info("Loading from cache...")
                datasets = {}
                for file in cache_dir.glob("*.csv"):
                    datasets[file.stem] = pd.read_csv(file)
                return datasets
            raise
    
    def load_all_datasets(self) -> Dict[str, any]:
        """
        Load all datasets
        
        Returns:
            Dictionary containing all datasets
        """
        logger.info("="*60)
        logger.info("Loading All Datasets for Predictive Maintenance")
        logger.info("="*60)
        
        all_data = {}
        
        # Load each dataset
        try:
            X_uci, y_uci = self.load_uci_ai4i()
            all_data['uci_ai4i'] = {'features': X_uci, 'targets': y_uci}
        except Exception as e:
            logger.error(f"Failed to load UCI AI4I: {e}")
        
        try:
            all_data['machine_failure'] = self.load_machine_failure()
        except Exception as e:
            logger.error(f"Failed to load Machine Failure: {e}")
        
        try:
            all_data['nasa_cmaps'] = self.load_nasa_cmaps()
        except Exception as e:
            logger.error(f"Failed to load NASA CMAPS: {e}")
        
        try:
            all_data['azure_maintenance'] = self.load_azure_maintenance()
        except Exception as e:
            logger.error(f"Failed to load Azure Maintenance: {e}")
        
        logger.info("="*60)
        logger.info(f"Successfully loaded {len(all_data)} dataset groups")
        logger.info("="*60)
        
        return all_data
    
    def get_dataset_info(self) -> pd.DataFrame:
        """
        Get summary information about all datasets
        
        Returns:
            DataFrame with dataset statistics
        """
        info = []
        
        all_data = self.load_all_datasets()
        
        for dataset_name, data in all_data.items():
            if isinstance(data, dict):
                if 'features' in data:
                    # UCI format
                    info.append({
                        'Dataset': dataset_name,
                        'Type': 'Separated Features/Targets',
                        'Samples': len(data['features']),
                        'Features': len(data['features'].columns),
                        'Has_Target': True
                    })
                else:
                    # Multi-table format (NASA, Azure)
                    for table_name, df in data.items():
                        info.append({
                            'Dataset': f"{dataset_name}/{table_name}",
                            'Type': 'Multi-table',
                            'Samples': len(df),
                            'Features': len(df.columns),
                            'Has_Target': 'failure' in df.columns or 'label' in df.columns
                        })
            elif isinstance(data, pd.DataFrame):
                info.append({
                    'Dataset': dataset_name,
                    'Type': 'Single Table',
                    'Samples': len(data),
                    'Features': len(data.columns),
                    'Has_Target': any(col.lower() in ['failure', 'target', 'label'] for col in data.columns)
                })
        
        return pd.DataFrame(info)


if __name__ == "__main__":
    # Test the data loader
    loader = DatasetLoader()
    
    # Load all datasets
    datasets = loader.load_all_datasets()
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    info_df = loader.get_dataset_info()
    print(info_df.to_string(index=False))
    print("="*60)
