"""
Prediction Module for Predictive Maintenance System
Generates failure predictions with interpretability
"""

import pandas as pd
import numpy as np
import joblib
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailurePredictor:
    """Production-ready failure prediction system"""
    
    def __init__(self, model_path: str = None, config_path: str = "config.yaml"):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            config_path: Path to config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = joblib.load(model_path)
        logger.info(f"✓ Loaded model from {model_path}")
    
    def predict_failure_probability(self, sensor_data: pd.DataFrame) -> np.ndarray:
        """
        Predict failure probability
        
        Args:
            sensor_data: DataFrame with sensor readings
            
        Returns:
            Array of probabilities (0-100%)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(sensor_data)[:, 1] * 100
        else:
            # For models without probability (e.g., Isolation Forest)
            predictions = self.model.predict(sensor_data)
            probabilities = (predictions == 1).astype(float) * 100
        
        return probabilities
    
    def predict_with_confidence(self, sensor_data: pd.DataFrame) -> Dict:
        """
        Predict with confidence interval
        
        Args:
            sensor_data: DataFrame with sensor readings
            
        Returns:
            Dictionary with predictions and confidence
        """
        probabilities = self.predict_failure_probability(sensor_data)
        
        # Calculate confidence based on probability distance from 0.5
        confidence = np.abs(probabilities - 50) / 50  # 0-1 scale
        
        results = {
            'failure_probability': probabilities,
            'confidence': confidence,
            'risk_level': self._determine_risk_level(probabilities)
        }
        
        return results
    
    def _determine_risk_level(self, probabilities: np.ndarray) -> np.ndarray:
        """Determine risk level based on probability thresholds"""
        thresholds = self.config['reports']['risk_thresholds']
        
        risk_levels = []
        for prob in probabilities:
            prob_decimal = prob / 100.0
            if prob_decimal >= thresholds['critical']:
                risk_levels.append('CRITICAL')
            elif prob_decimal >= thresholds['high']:
                risk_levels.append('HIGH')
            elif prob_decimal >= thresholds['medium']:
                risk_levels.append('MEDIUM')
            else:
                risk_levels.append('LOW')
        
        return np.array(risk_levels)
    
    def predict_time_to_failure(self, sensor_data: pd.DataFrame,
                               window_days: int = 7) -> Dict:
        """
        Predict time to failure within window
        
        Args:
            sensor_data: DataFrame with sensor readings
            window_days: Prediction window in days
            
        Returns:
            Dictionary with time predictions
        """
        probabilities = self.predict_failure_probability(sensor_data)
        
        # Estimate days to failure based on probability
        # Higher probability = sooner failure
        estimated_days = []
        for prob in probabilities:
            if prob >= 80:
                days = 1
            elif prob >= 60:
                days = 3
            elif prob >= 40:
                days = 5
            else:
                days = 7
            estimated_days.append(days)
        
        results = {
            'estimated_days_to_failure': np.array(estimated_days),
            'prediction_window': window_days,
            'probability': probabilities
        }
        
        return results
    
    def identify_root_causes(self, sensor_data: pd.DataFrame,
                            feature_names: list,
                            top_k: int = 5) -> Dict:
        """
        Identify root causes using feature importance or SHAP
        
        Args:
            sensor_data: DataFrame with sensor readings
            feature_names: List of feature names
            top_k: Number of top causes to return
            
        Returns:
            Dictionary with root causes
        """
        # Try to use SHAP if available
        try:
            import shap
            
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(sensor_data)
            
            # Get absolute SHAP values for each sample
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Get top contributing features for each sample
            root_causes_list = []
            for i in range(len(sensor_data)):
                sample_shap = np.abs(shap_values[i])
                top_indices = np.argsort(sample_shap)[-top_k:][::-1]
                
                causes = []
                for idx in top_indices:
                    causes.append({
                        'feature': feature_names[idx],
                        'value': sensor_data.iloc[i, idx],
                        'importance': float(sample_shap[idx])
                    })
                root_causes_list.append(causes)
            
            logger.info("✓ Root causes identified using SHAP")
            
        except Exception as e:
            logger.warning(f"SHAP not available: {e}. Using feature importance.")
            
            # Fallback to feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                top_indices = np.argsort(importance)[-top_k:][::-1]
                
                root_causes_list = []
                for i in range(len(sensor_data)):
                    causes = []
                    for idx in top_indices:
                        causes.append({
                            'feature': feature_names[idx],
                            'value': sensor_data.iloc[i, idx],
                            'importance': float(importance[idx])
                        })
                    root_causes_list.append(causes)
            else:
                root_causes_list = [[] for _ in range(len(sensor_data))]
        
        return {'root_causes': root_causes_list}
    
    def get_maintenance_recommendation(self, prediction_result: Dict) -> str:
        """
        Generate maintenance recommendation
        
        Args:
            prediction_result: Prediction results dictionary
            
        Returns:
            Maintenance recommendation string
        """
        risk_level = prediction_result.get('risk_level', 'UNKNOWN')
        probability = prediction_result.get('failure_probability', 0)
        
        recommendations = {
            'CRITICAL': "⚠️ IMMEDIATE ACTION REQUIRED:\n"
                      "- Stop machine operation immediately\n"
                      "- Conduct emergency inspection\n"
                      "- Replace critical components\n"
                      "- Do not restart until cleared by maintenance team",
            
            'HIGH': "🔴 URGENT MAINTENANCE NEEDED:\n"
                   "- Schedule maintenance within 24 hours\n"
                   "- Inspect all sensors showing abnormal readings\n"
                   "- Prepare replacement parts\n"
                   "- Monitor continuously until maintenance",
            
            'MEDIUM': "🟡 SCHEDULE PREVENTIVE MAINTENANCE:\n"
                     "- Plan maintenance within 3-5 days\n"
                     "- Monitor sensor trends closely\n"
                     "- Order necessary parts\n"
                     "- Continue normal operation with caution",
            
            'LOW': "🟢 CONTINUE NORMAL OPERATION:\n"
                  "- Maintain regular monitoring schedule\n"
                  "- No immediate action required\n"
                  "- Follow standard maintenance schedule"
        }
        
        return recommendations.get(risk_level, "Unable to determine recommendation")
    
    def predict_comprehensive(self, sensor_data: pd.DataFrame,
                             feature_names: list,
                             machine_id: str = "Unknown") -> Dict:
        """
        Comprehensive prediction with all information
        
        Args:
            sensor_data: DataFrame with sensor readings
            feature_names: List of feature names
            machine_id: Machine identifier
            
        Returns:
            Complete prediction report
        """
        logger.info(f"Generating comprehensive prediction for {machine_id}...")
        
        # Basic predictions
        prob_result = self.predict_with_confidence(sensor_data)
        time_result = self.predict_time_to_failure(sensor_data)
        root_causes = self.identify_root_causes(sensor_data, feature_names)
        
        # For single sample, extract first element
        if len(sensor_data) == 1:
            comprehensive_result = {
                'machine_id': machine_id,
                'timestamp': datetime.now().isoformat(),
                'failure_probability': float(prob_result['failure_probability'][0]),
                'confidence': float(prob_result['confidence'][0]),
                'risk_level': str(prob_result['risk_level'][0]),
                'estimated_days_to_failure': int(time_result['estimated_days_to_failure'][0]),
                'root_causes': root_causes['root_causes'][0] if root_causes['root_causes'] else [],
                'maintenance_recommendation': self.get_maintenance_recommendation({
                    'risk_level': prob_result['risk_level'][0],
                    'failure_probability': prob_result['failure_probability'][0]
                })
            }
        else:
            # Batch predictions
            comprehensive_result = {
                'machine_id': machine_id,
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(sensor_data),
                'predictions': []
            }
            
            for i in range(len(sensor_data)):
                comprehensive_result['predictions'].append({
                    'failure_probability': float(prob_result['failure_probability'][i]),
                    'confidence': float(prob_result['confidence'][i]),
                    'risk_level': str(prob_result['risk_level'][i]),
                    'estimated_days_to_failure': int(time_result['estimated_days_to_failure'][i]),
                    'root_causes': root_causes['root_causes'][i] if i < len(root_causes['root_causes']) else []
                })
        
        logger.info("✓ Comprehensive prediction complete")
        return comprehensive_result


if __name__ == "__main__":
    # Test the predictor
    predictor = FailurePredictor()
    logger.info("FailurePredictor initialized successfully")
