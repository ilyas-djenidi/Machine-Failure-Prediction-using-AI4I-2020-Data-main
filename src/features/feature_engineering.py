"""
Feature Engineering Module for Predictive Maintenance
Creates advanced features for failure prediction
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import PolynomialFeatures
import logging
import yaml
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for predictive maintenance"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               windows: List[int] = None,
                               columns: List[str] = None) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: Input DataFrame (must be sorted by time)
            windows: List of window sizes
            columns: Columns to create rolling features for
            
        Returns:
            DataFrame with added rolling features
        """
        if windows is None:
            windows = self.feature_config['rolling_windows']
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Creating rolling features with windows: {windows}")
        
        for col in columns:
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling min
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).min()
                
                # Rolling max
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).max()
                
                # Exponential moving average
                df[f'{col}_ema_{window}'] = df[col].ewm(
                    span=window, adjust=False
                ).mean()
        
        logger.info(f"Added {len(windows) * len(columns) * 5} rolling features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame,
                           lags: List[int] = None,
                           columns: List[str] = None) -> pd.DataFrame:
        """
        Create lag features (previous values)
        
        Args:
            df: Input DataFrame
            lags: List of lag periods
            columns: Columns to create lag features for
            
        Returns:
            DataFrame with added lag features
        """
        if lags is None:
            lags = self.feature_config['lag_features']
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Creating lag features with lags: {lags}")
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Fill NaN values created by lag features
        df = df.fillna(method='bfill').fillna(0)
        
        logger.info(f"Added {len(lags) * len(columns)} lag features")
        return df
    
    def create_trend_features(self, df: pd.DataFrame,
                             columns: List[str] = None) -> pd.DataFrame:
        """
        Create trend and change features
        
        Args:
            df: Input DataFrame
            columns: Columns to create trend features for
            
        Returns:
            DataFrame with added trend features
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info("Creating trend features...")
        
        for col in columns:
            # Rate of change
            df[f'{col}_rate_of_change'] = df[col].pct_change()
            
            # Absolute change
            df[f'{col}_diff'] = df[col].diff()
            
            # Acceleration (second derivative)
            df[f'{col}_acceleration'] = df[col].diff().diff()
            
            # Linear trend over last 7 periods
            df[f'{col}_trend_7'] = df[col].rolling(window=7, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=True
            )
        
        # Fill infinities and NaNs
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        logger.info(f"Added {len(columns) * 4} trend features")
        return df
    
    def create_degradation_indicators(self, df: pd.DataFrame,
                                      baseline_window: int = 30) -> pd.DataFrame:
        """
        Create degradation indicators
        
        Args:
            df: Input DataFrame
            baseline_window: Window for baseline calculation
            
        Returns:
            DataFrame with degradation indicators
        """
        logger.info("Creating degradation indicators...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            # Deviation from baseline (moving average)
            baseline = df[col].rolling(window=baseline_window, min_periods=1).mean()
            df[f'{col}_deviation_from_baseline'] = df[col] - baseline
            
            # Cumulative deviation
            df[f'{col}_cumulative_deviation'] = df[f'{col}_deviation_from_baseline'].cumsum()
            
            # Distance from normal range (assuming data is normalized)
            df[f'{col}_abs_deviation'] = abs(df[col])
        
        logger.info(f"Added {len(numeric_cols) * 3} degradation indicators")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                    degree: int = None) -> pd.DataFrame:
        """
        Create polynomial and interaction features
        
        Args:
            df: Input DataFrame
            degree: Polynomial degree
            
        Returns:
            DataFrame with interaction features
        """
        if not self.feature_config.get('include_interactions', True):
            return df
        
        if degree is None:
            degree = self.feature_config.get('max_interaction_degree', 2)
        
        logger.info(f"Creating interaction features (degree={degree})...")
        
        # Select only a subset of important features to avoid explosion
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to prevent memory issues
        if len(numeric_cols) > 10:
            logger.warning(f"Too many columns ({len(numeric_cols)}), selecting top 10 for interactions")
            numeric_cols = numeric_cols[:10]
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        interactions = poly.fit_transform(df[numeric_cols])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Add interaction features
        for i, name in enumerate(feature_names):
            if name not in numeric_cols:  # Only add new interactions
                df[name] = interactions[:, i]
        
        logger.info(f"Added {len(feature_names) - len(numeric_cols)} interaction features")
        return df
    
    def create_time_to_failure(self, df: pd.DataFrame,
                               failure_col: str,
                               time_col: str = None) -> pd.DataFrame:
        """
        Create time-to-failure features
        
        Args:
            df: Input DataFrame
            failure_col: Column indicating failure (1=failure, 0=normal)
            time_col: Time column (optional)
            
        Returns:
            DataFrame with time-to-failure features
        """
        logger.info("Creating time-to-failure features...")
        
        # Create time index if not provided
        if time_col is None:
            df = df.reset_index(drop=True)
            time_index = df.index
        else:
            time_index = df[time_col]
        
        # Calculate cycles/days until failure
        df['cycles_to_failure'] = 0
        failure_indices = df[df[failure_col] == 1].index
        
        for fail_idx in failure_indices:
            # Set cycles to failure for all rows before this failure
            mask = (df.index < fail_idx)
            df.loc[mask, 'cycles_to_failure'] = fail_idx - df.loc[mask].index
        
        # Binary flags for different prediction windows
        for window in self.config['models']['prediction_windows']:
            df[f'failure_within_{window}d'] = (
                (df['cycles_to_failure'] > 0) & (df['cycles_to_failure'] <= window)
            ).astype(int)
        
        logger.info(f"Added {1 + len(self.config['models']['prediction_windows'])} time-to-failure features")
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       k: int = None) -> pd.DataFrame:
        """
        Select top K features using mutual information
        
        Args:
            X: Features DataFrame
            y: Target Series
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if not self.feature_config.get('feature_selection', False):
            return X
        
        if k is None:
            k = self.feature_config.get('feature_selection_k', 50)
        
        if X.shape[1] <= k:
            logger.info(f"Number of features ({X.shape[1]}) <= k ({k}), keeping all features")
            return X
        
        logger.info(f"Selecting top {k} features from {X.shape[1]} using mutual information...")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        logger.info(f"Selected features: {selected_features[:10]}...")  # Show first 10
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def full_pipeline(self, df: pd.DataFrame,
                     failure_col: str = None,
                     time_col: str = None) -> pd.DataFrame:
        """
        Run full feature engineering pipeline
        
        Args:
            df: Input DataFrame
            failure_col: Failure indicator column (optional)
            time_col: Time column (optional)
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("="*60)
        logger.info("Running Full Feature Engineering Pipeline")
        logger.info("="*60)
        
        original_features = df.shape[1]
        
        # Rolling features
        df = self.create_rolling_features(df)
        
        # Lag features
        df = self.create_lag_features(df)
        
        # Trend features
        df = self.create_trend_features(df)
        
        # Degradation indicators
        df = self.create_degradation_indicators(df)
        
        # Interaction features (expensive, optional)
        if self.feature_config.get('include_interactions', False):
            df = self.create_interaction_features(df)
        
        # Time to failure (if failure column provided)
        if failure_col and failure_col in df.columns:
            df = self.create_time_to_failure(df, failure_col, time_col)
        
        final_features = df.shape[1]
        logger.info(f"\n{'='*60}")
        logger.info(f"Feature Engineering Complete!")
        logger.info(f"Features: {original_features} → {final_features} (+{final_features - original_features})")
        logger.info(f"{'='*60}\n")
        
        return df


if __name__ == "__main__":
    # Test the feature engineer
    engineer = FeatureEngineer()
    logger.info("FeatureEngineer initialized successfully")
