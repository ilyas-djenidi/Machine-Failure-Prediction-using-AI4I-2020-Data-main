"""
Model Training Module for Predictive Maintenance System
Trains multiple models and handles hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score
import joblib
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Comprehensive model training framework"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']
        self.models = {}
        self.best_params = {}
        
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  tune: bool = False) -> LogisticRegression:
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            tune: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info("Training Logistic Regression...")
        
        if tune:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            model = LogisticRegression(random_state=self.config['preprocessing']['random_state'])
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            self.best_params['logistic_regression'] = grid_search.best_params_
            logger.info(f"Best params: {grid_search.best_params_}")
        else:
            model = LogisticRegression(
                random_state=self.config['preprocessing']['random_state'],
                max_iter=1000
            )
            model.fit(X_train, y_train)
        
        logger.info("✓ Logistic Regression trained")
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           tune: bool = False) -> RandomForestClassifier:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
            tune: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest...")
        
        rf_config = self.model_config.get('random_forest', {})
        
        if tune:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            model = RandomForestClassifier(random_state=rf_config.get('random_state', 42))
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            self.best_params['random_forest'] = grid_search.best_params_
            logger.info(f"Best params: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=rf_config.get('n_estimators', 200),
                max_depth=rf_config.get('max_depth', 20),
                min_samples_split=rf_config.get('min_samples_split', 5),
                random_state=rf_config.get('random_state', 42),
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        logger.info("✓ Random Forest trained")
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     tune: bool = False) -> XGBClassifier:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            tune: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost...")
        
        xgb_config = self.model_config.get('xgboost', {})
        
        if tune:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            model = XGBClassifier(random_state=xgb_config.get('random_state', 42))
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            self.best_params['xgboost'] = grid_search.best_params_
            logger.info(f"Best params: {grid_search.best_params_}")
        else:
            model = XGBClassifier(
                n_estimators=xgb_config.get('n_estimators', 300),
                max_depth=xgb_config.get('max_depth', 10),
                learning_rate=xgb_config.get('learning_rate', 0.1),
                subsample=xgb_config.get('subsample', 0.8),
                colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
                random_state=xgb_config.get('random_state', 42),
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        logger.info("✓ XGBoost trained")
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      tune: bool = False) -> LGBMClassifier:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            tune: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info("Training LightGBM...")
        
        lgbm_config = self.model_config.get('lightgbm', {})
        
        if tune:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70]
            }
            
            model = LGBMClassifier(random_state=lgbm_config.get('random_state', 42), verbose=-1)
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            self.best_params['lightgbm'] = grid_search.best_params_
            logger.info(f"Best params: {grid_search.best_params_}")
        else:
            model = LGBMClassifier(
                n_estimators=lgbm_config.get('n_estimators', 300),
                max_depth=lgbm_config.get('max_depth', 10),
                learning_rate=lgbm_config.get('learning_rate', 0.1),
                num_leaves=lgbm_config.get('num_leaves', 31),
                random_state=lgbm_config.get('random_state', 42),
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        logger.info("✓ LightGBM trained")
        return model
    
    def train_isolation_forest(self, X_train: pd.DataFrame) -> IsolationForest:
        """
        Train Isolation Forest for anomaly detection
        
        Args:
            X_train: Training features (unsupervised)
            
        Returns:
            Trained model
        """
        logger.info("Training Isolation Forest (Anomaly Detector)...")
        
        iso_config = self.model_config.get('isolation_forest', {})
        
        model = IsolationForest(
            contamination=iso_config.get('contamination', 0.1),
            random_state=iso_config.get('random_state', 42),
            n_jobs=-1
        )
        model.fit(X_train)
        
        logger.info("✓ Isolation Forest trained")
        return model
    
    def train_all_baselines(self, X_train: pd.DataFrame, y_train: pd.Series,
                           tune: bool = False) -> Dict[str, Any]:
        """
        Train all baseline models
        
        Args:
            X_train: Training features
            y_train: Training targets
            tune: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of trained models
        """
        logger.info("="*60)
        logger.info("Training All Baseline Models")
        logger.info("="*60)
        
        models = {}
        
        # Train each model
        models['logistic_regression'] = self.train_logistic_regression(X_train, y_train, tune)
        models['random_forest'] = self.train_random_forest(X_train, y_train, tune)
        models['xgboost'] = self.train_xgboost(X_train, y_train, tune)
        models['lightgbm'] = self.train_lightgbm(X_train, y_train, tune)
        models['isolation_forest'] = self.train_isolation_forest(X_train)
        
        self.models = models
        
        logger.info("="*60)
        logger.info(f"Trained {len(models)} models successfully")
        logger.info("="*60)
        
        return models
    
    def save_model(self, model: Any, name: str, output_dir: str = "models"):
        """
        Save model to disk
        
        Args:
            model: Trained model
            name: Model name
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"{name}_{timestamp}.pkl"
        
        joblib.dump(model, filename)
        logger.info(f"✓ Saved {name} to {filename}")
        
        # Also save as latest
        latest_filename = output_path / f"{name}_latest.pkl"
        joblib.dump(model, latest_filename)
        logger.info(f"✓ Saved {name} to {latest_filename}")
    
    def save_all_models(self, models: Dict[str, Any] = None, output_dir: str = "models"):
        """Save all trained models"""
        if models is None:
            models = self.models
        
        logger.info(f"Saving {len(models)} models...")
        for name, model in models.items():
            self.save_model(model, name, output_dir)
        
        # Save hyperparameters
        if self.best_params:
            params_file = Path(output_dir) / "best_hyperparameters.yaml"
            with open(params_file, 'w') as f:
                yaml.dump(self.best_params, f)
            logger.info(f"✓ Saved hyperparameters to {params_file}")
    
    def load_model(self, name: str, input_dir: str = "models") -> Any:
        """Load a saved model"""
        filepath = Path(input_dir) / f"{name}_latest.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"✓ Loaded {name} from {filepath}")
        return model


if __name__ == "__main__":
    # Test the trainer
    trainer = ModelTrainer()
    logger.info("ModelTrainer initialized successfully")
