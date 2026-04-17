"""
Data Preprocessing Module for Predictive Maintenance System
Handles missing values, outliers, normalization, and class balancing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
import logging
import yaml
from pathlib import Path
from typing import Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor with configuration
        
        Args:
            config_path: Path to config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config['preprocessing']
        self.scaler = None
        self.imputer = None
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = None) -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            df: Input DataFrame
            strategy: 'forward_fill', 'mean', 'median', or 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        if strategy is None:
            strategy = self.preprocessing_config['missing_value_strategy']
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        missing_before = df.isnull().sum().sum()
        
        if strategy == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        elif strategy == 'median':
            self.imputer = SimpleImputer(strategy='median')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        elif strategy == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} → {missing_after}")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: list = None, 
                       method: str = None) -> pd.DataFrame:
        """
        Remove outliers from numeric columns
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (None = all numeric)
            method: 'iqr', 'zscore', or 'isolation_forest'
            
        Returns:
            DataFrame with outliers removed
        """
        if method is None:
            method = self.preprocessing_config['outlier_method']
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Removing outliers using {method} method...")
        rows_before = len(df)
        
        if method == 'iqr':
            threshold = self.preprocessing_config.get('outlier_threshold', 1.5)
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            threshold = self.preprocessing_config.get('outlier_threshold', 3.0)
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df = df[z_scores < threshold]
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=self.preprocessing_config['random_state']
            )
            outliers = iso_forest.fit_predict(df[columns])
            df = df[outliers != -1]
        
        rows_after = len(df)
        logger.info(f"Rows: {rows_before} → {rows_after} (removed {rows_before - rows_after})")
        
        return df
    
    def normalize_features(self, X: pd.DataFrame, method: str = None, 
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize/scale features
        
        Args:
            X: Features DataFrame
            method: 'standard', 'minmax', or 'robust'
            fit: Whether to fit the scaler (True) or just transform (False)
            
        Returns:
            Normalized DataFrame
        """
        if method is None:
            method = self.preprocessing_config['normalization']
        
        logger.info(f"Normalizing features using {method} scaling...")
        
        if fit or self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            logger.info(f"Encoding columns: {list(categorical_cols)}")
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df
    
    def balance_classes(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance class distribution
        
        Args:
            X: Features DataFrame
            y: Target Series
            method: 'smote', 'random_oversample', 'random_undersample', or 'class_weights'
            
        Returns:
            Tuple of (balanced X, balanced y)
        """
        if method is None:
            method = self.preprocessing_config.get('balance_method', 'smote')
        
        if method == 'class_weights':
            logger.info("Using class weights (no resampling)")
            return X, y
        
        logger.info(f"Balancing classes using {method}...")
        class_dist_before = y.value_counts().to_dict()
        logger.info(f"Distribution before: {class_dist_before}")
        
        if method == 'smote':
            try:
                sampler = SMOTE(random_state=self.preprocessing_config['random_state'])
                X_balanced, y_balanced = sampler.fit_resample(X, y)
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Using RandomOverSampler instead.")
                sampler = RandomOverSampler(random_state=self.preprocessing_config['random_state'])
                X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        elif method == 'random_oversample':
            sampler = RandomOverSampler(random_state=self.preprocessing_config['random_state'])
            X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        elif method == 'random_undersample':
            sampler = RandomUnderSampler(random_state=self.preprocessing_config['random_state'])
            X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        else:
            raise ValueError(f"Unknown balance method: {method}")
        
        class_dist_after = pd.Series(y_balanced).value_counts().to_dict()
        logger.info(f"Distribution after: {class_dist_after}")
        
        return X_balanced, y_balanced
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series,
                               test_size: float = None,
                               val_size: float = None) -> Dict:
        """
        Create train/validation/test splits
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Dictionary with train/val/test splits
        """
        if test_size is None:
            test_size = self.preprocessing_config['test_size']
        if val_size is None:
            val_size = self.preprocessing_config.get('validation_size', 0.1)
        
        logger.info(f"Creating splits: train={1-test_size-val_size:.1%}, "
                   f"val={val_size:.1%}, test={test_size:.1%}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.preprocessing_config['random_state'],
            stratify=y
        )
        
        # Second split: separate validation set from training
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.preprocessing_config['random_state'],
                stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        splits = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val) if X_val is not None else 0}, "
                   f"Test size: {len(X_test)}")
        
        return splits
    
    def full_pipeline(self, df: pd.DataFrame, target_col: str,
                     balance: bool = True) -> Dict:
        """
        Run full preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            balance: Whether to balance classes
            
        Returns:
            Dictionary with processed data splits
        """
        logger.info("="*60)
        logger.info("Running Full Preprocessing Pipeline")
        logger.info("="*60)
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Encode categorical variables
        X = self.encode_categorical(X)
        
        # Remove outliers
        X = self.remove_outliers(X)
        
        # Normalize features
        X = self.normalize_features(X, fit=True)
        
        # Create train/test splits
        splits = self.create_train_test_split(X, y)
        
        # Balance training set if requested
        if balance:
            splits['X_train'], splits['y_train'] = self.balance_classes(
                splits['X_train'],
                splits['y_train']
            )
        
        logger.info("="*60)
        logger.info("Preprocessing Complete!")
        logger.info("="*60)
        
        return splits
    
    def save_splits(self, splits: Dict, output_dir: str = "data/processed"):
        """Save processed data splits to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, data in splits.items():
            if data is not None:
                filepath = output_path / f"{name}.csv"
                data.to_csv(filepath, index=False)
                logger.info(f"Saved {name} to {filepath}")


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    logger.info("DataPreprocessor initialized successfully")
