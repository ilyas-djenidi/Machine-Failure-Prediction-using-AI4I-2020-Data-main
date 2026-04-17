"""
Model Evaluation Module for Predictive Maintenance System
Comprehensive evaluation metrics and visualization
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config['evaluation']
        self.results = {}
    
    def evaluate_classification(self, model: Any, X_test: pd.DataFrame, 
                               y_test: pd.Series, model_name: str = "model") -> Dict:
        """
        Evaluate classification model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        # ROC-AUC (only if probabilities available)
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['roc_auc'] = None
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        # Log metrics
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, models: Dict[str, Any], 
                           X_test: pd.DataFrame, y_test: pd.Series) -> pd. DataFrame:
        """
        Evaluate multiple models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with all model metrics
        """
        logger.info("="*60)
        logger.info("Evaluating All Models")
        logger.info("="*60)
        
        all_metrics = []
        
        for name, model in models.items():
            try:
                metrics = self.evaluate_classification(model, X_test, y_test, name)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        results_df = pd.DataFrame(all_metrics)
        results_df = results_df.sort_values('f1', ascending=False)
        
        logger.info("\n" + "="*60)
        logger.info("Model Comparison")
        logger.info("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "Confusion Matrix",
                             save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved confusion matrix to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, models_data: Dict[str, Dict] = None,
                      save_path: str = None):
        """
        Plot ROC curves for multiple models
        
        Args:
            models_data: Dictionary with model results
            save_path: Path to save figure
        """
        if models_data is None:
            models_data = self.results
        
        plt.figure(figsize=(10, 8))
        
        for name, data in models_data.items():
            if 'y_true' in data and 'y_proba' in data:
                try:
                    fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
                except:
                    pass
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved ROC curve to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    title: str = "Precision-Recall Curve",
                                    save_path: str = None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved PR curve to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def feature_importance(self, model: Any, feature_names: List[str],
                          top_k: int = 20, save_path: str = None) -> pd.DataFrame:
        """
        Extract and plot feature importance
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_k: Number of top features to show
            save_path: Path to save figure
            
        Returns:
            DataFrame with feature importances
        """
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not have feature importance attribute")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top K features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_k)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved feature importance to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return importance_df
    
    def check_performance_thresholds(self, metrics: Dict) -> bool:
        """
        Check if model meets minimum performance thresholds
        
        Args:
            metrics: Dictionary of model metrics
            
        Returns:
            True if all thresholds are met
        """
        min_precision = self.eval_config.get('min_precision', 0.80)
        min_recall = self.eval_config.get('min_recall', 0.75)
        min_f1 = self.eval_config.get('min_f1', 0.77)
        
        passed = True
        
        if metrics['precision'] < min_precision:
            logger.warning(f"⚠ Precision ({metrics['precision']:.3f}) below threshold ({min_precision})")
            passed = False
        
        if metrics['recall'] < min_recall:
            logger.warning(f"⚠ Recall ({metrics['recall']:.3f}) below threshold ({min_recall})")
            passed = False
        
        if metrics['f1'] < min_f1:
            logger.warning(f"⚠ F1 ({metrics['f1']:.3f}) below threshold ({min_f1})")
            passed = False
        
        if passed:
            logger.info("✓ All performance thresholds met!")
        
        return passed
    
    def save_results(self, results_df: pd.DataFrame, output_dir: str = "reports"):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_path / "model_metrics.csv"
        results_df.to_csv(metrics_file, index=False)
        logger.info(f"✓ Saved metrics to {metrics_file}")
        
        # Save detailed results
        for name, data in self.results.items():
            if 'y_true' in data and 'y_pred' in data:
                # Classification report
                report = classification_report(data['y_true'], data['y_pred'])
                report_file = output_path / f"{name}_classification_report.txt"
                with open(report_file, 'w') as f:
                    f.write(report)


if __name__ == "__main__":
    # Test the evaluator
    evaluator = ModelEvaluator()
    logger.info("ModelEvaluator initialized successfully")
